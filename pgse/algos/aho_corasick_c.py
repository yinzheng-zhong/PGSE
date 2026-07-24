import ctypes
import os

import numpy as np
import sys

from typing import TYPE_CHECKING, Optional, Union

from pgse.algos.aho_corasick_base import AhoCorasickBase
from pgse.etc.alphabet import Alphabet, get_alphabet
from pgse.log import logger

if TYPE_CHECKING:
    # Imported lazily at runtime in _get_fallback to avoid a circular import.
    from pgse.algos.aho_corasick_py import AhoCorasickPy

BYTE_VALUES = 256

# The byte tables, once marshalled into ctypes: the char -> index array, the alphabet
# size, and the optional complement buffer.
Tables = tuple[
    'ctypes.Array[ctypes.c_int]',
    int,
    Optional['ctypes.Array[ctypes.c_char]'],
]


class AhoCorasickC(AhoCorasickBase):
    def __init__(self) -> None:
        super().__init__()
        self.lib: ctypes.CDLL = self._load_lib()
        self._fallback: Optional['AhoCorasickPy'] = None
        self._tables_cache: tuple[Optional[Alphabet], Optional[Tables]] = (None, None)

    def _load_lib(self) -> ctypes.CDLL:
        # Determine the library name based on the platform
        if sys.platform.startswith('linux'):
            lib_name = 'aho_corasick.so'
        elif sys.platform.startswith('darwin'):
            lib_name = 'aho_corasick.dylib'
        elif sys.platform.startswith('win32'):
            lib_name = 'aho_corasick.dll'
        else:
            raise RuntimeError("Unsupported platform")

        # Get the path relative to this module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, '..', 'c_lib', lib_name)

        try:
            lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise FileNotFoundError(f"Could not load shared library at {lib_path}: {e}")

        if not hasattr(lib, 'count_segments_ex'):
            # A library compiled before arbitrary alphabets were supported. It can only
            # count DNA, so recompile c_lib to use anything else.
            raise FileNotFoundError(
                f"The shared library at {lib_path} predates alphabet support. Recompile it from c_lib."
            )

        # Define function signature
        lib.count_segments_ex.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # nodes
            ctypes.c_int,                     # num_nodes
            ctypes.POINTER(ctypes.c_char_p),  # segments
            ctypes.c_int,                     # num_segments
            ctypes.POINTER(ctypes.c_int),     # result_counts
            ctypes.POINTER(ctypes.c_int),     # char_index (256 entries)
            ctypes.c_int,                     # alphabet_size
            ctypes.POINTER(ctypes.c_char)     # complement table (256 entries) or NULL
        ]
        lib.count_segments_ex.restype = None

        return lib

    def _get_tables(self, alphabet: Alphabet) -> Tables:
        """
        Build (and cache) the byte tables the C library needs for the given alphabet.

        :param alphabet: Alphabet: The alphabet in use.
        :return: tuple: (char_index array, alphabet size, complement buffer or None).
        """
        cached_alphabet, cached_tables = self._tables_cache
        if cached_alphabet == alphabet and cached_tables is not None:
            return cached_tables

        index, size, complement = alphabet.byte_tables()
        # Create ctypes arrays for the index and complement
        tables: Tables = (
            (ctypes.c_int * BYTE_VALUES)(*index),
            size,
            ctypes.create_string_buffer(complement, BYTE_VALUES) if complement is not None else None
        )

        self._tables_cache = (alphabet, tables)
        return tables

    def _get_fallback(self) -> 'AhoCorasickPy':
        if self._fallback is None:
            from pgse.algos.aho_corasick_py import AhoCorasickPy
            self._fallback = AhoCorasickPy()
        return self._fallback

    def count_segments(self, nodes: list[str], segments: list[str]) -> Union[np.ndarray, list[int]]:
        """
        Count the number of segments in each node.
        :param nodes: list of nodes. Contigs/Scaffolds.
        :param segments: list of segments. Sequences to count.
        :return: A per-segment count array (or a list when the multi-byte Python
            fallback is used). Callers wrap the result in ``np.asarray``.
        """
        alphabet = get_alphabet()

        if not alphabet.is_byte_safe:
            # Multi-byte characters cannot be indexed by byte in the C library.
            logger.warning(
                f'{alphabet} contains multi-byte characters. Falling back to the slower Python implementation.'
            )
            return self._get_fallback().count_segments(nodes, segments)

        char_index, alphabet_size, complement = self._get_tables(alphabet)

        num_nodes = len(nodes)
        num_segments = len(segments)

        # Create arrays of c_char_p. latin-1 keeps one byte per character, which is what
        # the byte tables above are indexed by.
        node_array = (ctypes.c_char_p * num_nodes)(*(node.encode('latin-1') for node in nodes))
        segment_array = (ctypes.c_char_p * num_segments)(
            *(alphabet.normalise(seg).encode('latin-1') for seg in segments)
        )

        # Prepare result array
        result_counts = (ctypes.c_int * num_segments)()

        # Call the C function
        self.lib.count_segments_ex(
            node_array, ctypes.c_int(num_nodes),
            segment_array, ctypes.c_int(num_segments),
            result_counts,
            char_index, ctypes.c_int(alphabet_size),
            complement
        )

        # Convert result to NumPy array
        seq_count = np.ctypeslib.as_array(result_counts, shape=(num_segments,))

        return seq_count
