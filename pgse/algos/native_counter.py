"""Segment counting through the native Rust kernel (``pgse._native``).

The Rust :class:`SegmentMatcher` builds the Aho-Corasick automaton **once** and
searches every sequence against that single shared automaton with a rayon thread
pool.

The kernel returns CSR triplets ``(indptr, indices, data)`` per batch; this module
stitches them into a single SciPy CSR matrix (or a dense array when sparse storage
is off). Counts come back as ``uint32`` and are cast to the requested storage dtype
here, saturating at ``UINT16_MAX`` for the ``uint16`` option.
"""

from typing import Iterable, List, Optional, Sequence as Seq, Tuple, Union

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from pgse.etc.alphabet import Alphabet, get_alphabet
from pgse.log import logger

try:
    from pgse import _native  # type: ignore
    _HAVE_NATIVE = True
except ImportError:  # pragma: no cover - only where the extension was not built.
    _native = None  # type: ignore
    _HAVE_NATIVE = False
    # Installs are meant to build the extension (optional = false in pyproject), so
    # reaching here means something is wrong. Warn loudly rather than silently taking
    # the much slower Python path.
    logger.warning(
        'The native counting kernel (pgse._native) is not available; falling back to '
        'the much slower pure-Python counter. Install a Rust toolchain '
        '(https://rustup.rs) and reinstall PGSE to build it.'
    )

UINT16_MAX = int(np.iinfo(np.uint16).max)

Dataset = Union[np.ndarray, sp.csr_matrix]
Triplet = Tuple[np.ndarray, np.ndarray, np.ndarray]

# Number of samples handed to the kernel per call. Bounds the Python->Rust copy and
# gives a progress bar without materialising every sample's nodes at once.
DEFAULT_CHUNK_SIZE = 2048


def native_available() -> bool:
    """True when the compiled Rust counting kernel can be used."""
    return _HAVE_NATIVE


def complement_bytes(alphabet: Alphabet) -> Optional[bytes]:
    """
    The 256-entry reverse-complement table for the alphabet, or ``None`` when it has
    no complement (canonicalisation off). Matches the table the C library used.
    """
    _, _, complement = alphabet.byte_tables()
    return complement


def raw_matcher(norm_segments: Seq[str], complement: Optional[bytes]):
    """
    Build a matcher from already-normalised segments and a complement table.

    Used by the distributed path, which normalises the pool once on the driver and
    ships it to each node.
    """
    if not _HAVE_NATIVE:
        raise RuntimeError('The native counting kernel is not available.')
    return _native.SegmentMatcher(norm_segments, complement)


def build_matcher(segments: Seq[str], alphabet: Optional[Alphabet] = None):
    """
    Build a reusable automaton from the segment pool.

    Segments are case-normalised here (the kernel does no normalisation); node text
    is already normalised when sequences are read, so it is passed through as-is.
    """
    alphabet = alphabet or get_alphabet()
    norm_segments = [alphabet.normalise(s) for s in segments]
    return raw_matcher(norm_segments, complement_bytes(alphabet))


def assemble_from_triplets(
        triplets: Iterable[Triplet],
        n_rows: int,
        n_cols: int,
        dtype: np.dtype,
        sparse: bool,
) -> Dataset:
    """
    Concatenate per-batch CSR triplets into one matrix.

    Each triplet's ``indptr`` is batch-local (starts at 0); they are rebased onto a
    running offset. Counts are cast to ``dtype`` (saturated for uint16). Returns a
    CSR matrix when ``sparse`` else a dense ndarray.
    """
    indptr_parts: List[np.ndarray] = [np.zeros(1, dtype=np.int64)]
    indices_parts: List[np.ndarray] = []
    data_parts: List[np.ndarray] = []
    offset = 0
    for indptr, indices, data in triplets:
        # Drop the leading 0 of each local indptr and shift onto the running total.
        indptr_parts.append(indptr[1:].astype(np.int64, copy=False) + offset)
        offset += int(indptr[-1])
        indices_parts.append(indices)
        data_parts.append(data)

    full_indptr = np.concatenate(indptr_parts)
    indices = np.concatenate(indices_parts) if indices_parts else np.zeros(0, np.int32)
    data = np.concatenate(data_parts) if data_parts else np.zeros(0, np.uint32)

    if dtype == np.uint16:
        data = np.minimum(data, UINT16_MAX)
    data = data.astype(dtype, copy=False)

    csr = sp.csr_matrix((data, indices, full_indptr), shape=(n_rows, n_cols), dtype=dtype)
    return csr if sparse else csr.toarray()


def count_matrix(
        matcher,
        sequences: Seq,
        n_cols: int,
        dtype: np.dtype,
        sparse: bool,
        threads: int,
        desc: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dataset:
    """
    Count segment occurrences for every sequence against a prebuilt ``matcher``.

    Runs locally: the kernel parallelises across ``threads`` rayon workers. Sequences
    are processed in chunks purely for progress reporting and to bound the transient
    Python->Rust copy; the automaton is built once and reused across all chunks.
    """
    n_rows = len(sequences)
    alphabet = get_alphabet()
    triplets: List[Triplet] = []
    for start in tqdm(range(0, n_rows, chunk_size), desc=desc):
        chunk = sequences[start:start + chunk_size]
        # Normalise node text so counting matches the Python reference exactly. For a
        # case-sensitive alphabet ``normalise`` returns the string unchanged, so this
        # is effectively free; for a case-insensitive one it folds case, as the kernel
        # itself does no normalisation.
        node_lists = [normalised_nodes(seq, alphabet) for seq in chunk]
        triplets.append(matcher.count_csr(node_lists, threads))
    return assemble_from_triplets(triplets, n_rows, n_cols, dtype, sparse)


def normalised_nodes(seq, alphabet: Optional[Alphabet] = None) -> List[str]:
    """The sequence's node strings, case-folded to match the active alphabet."""
    alphabet = alphabet or get_alphabet()
    return [alphabet.normalise(node) for node in seq.nodes]
