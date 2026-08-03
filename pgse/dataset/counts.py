"""Assembly of per-sample segment counts into the count matrix."""

from typing import Iterable, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from tqdm import tqdm

# The count matrix can be stored either densely (default, np.float32) or as a
# uint16 array to halve the footprint.
UINT16_MAX = int(np.iinfo(np.uint16).max)

Dataset = Union[np.ndarray, sp.csr_matrix]


def assemble_counts(
        rows: Iterable[np.ndarray],
        n_rows: int,
        n_cols: int,
        dtype: npt.DTypeLike,
        sparse: bool,
        desc: str,
) -> Dataset:
    """
    Build the (n_rows x n_cols) segment-count matrix from per-sample count rows.

    :param rows: iterable yielding one count row (array-like, length n_cols) per
        sample, in order.
    :param dtype: storage dtype for the counts (np.float32 or np.uint16).
    :param sparse: if True, return a scipy CSR matrix instead of a dense ndarray.
    """
    if sparse:
        # Build CSR arrays directly; only one dense row is materialised at a time.
        indptr = np.empty(n_rows + 1, dtype=np.int64)
        indptr[0] = 0
        indices_chunks: list[np.ndarray] = []
        data_chunks: list[np.ndarray] = []
        for i, row in enumerate(tqdm(rows, total=n_rows, desc=desc)):
            nz = np.flatnonzero(row)
            vals = row[nz]
            if dtype == np.uint16:
                vals = np.minimum(vals, UINT16_MAX)
            indices_chunks.append(nz.astype(np.int32, copy=False))
            data_chunks.append(vals.astype(dtype, copy=False))
            indptr[i + 1] = indptr[i] + nz.size

        indices = np.concatenate(indices_chunks) if indices_chunks else np.zeros(0, np.int32)
        data = np.concatenate(data_chunks) if data_chunks else np.zeros(0, dtype)
        return sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols), dtype=dtype)

    # Dense path: preallocate once and fill row by row, so the list of per-sample
    # arrays and a second full copy are never held simultaneously.
    out = np.empty((n_rows, n_cols), dtype=dtype)
    for i, row in enumerate(tqdm(rows, total=n_rows, desc=desc)):
        if dtype == np.uint16:
            row = np.minimum(row, UINT16_MAX)
        out[i] = row
    return out
