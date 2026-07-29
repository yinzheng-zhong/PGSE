from typing import Iterable, Iterator, Optional, Union

import numpy as np
import scipy.sparse as sp

from pgse.dataset.file_label import FileLabel
from tqdm import tqdm

from pgse.etc.alphabet import Alphabet, get_alphabet, set_alphabet
from pgse.genome import seq_manager
from pgse.log import logger
from pgse.genome.sequence import Sequence
import ray
from pgse.segment import seg_pool
from pgse.segment.segment_pool import SegmentPool

# The count matrix can be stored either densely (default, np.float32) or as a
# uint16 array to halve the footprint. Counts are non-negative integers, so
# uint16 is lossless up to UINT16_MAX; anything larger is saturated rather than
# allowed to wrap silently.
UINT16_MAX = int(np.iinfo(np.uint16).max)

Dataset = Union[np.ndarray, sp.csr_matrix]


def _row_stream(tasks: list) -> Iterator[np.ndarray]:
    """
    Yield each Ray task's count row one at a time so the finished object can be
    released before the next is fetched, instead of holding all rows at once.
    """
    for task in tasks:
        yield np.asarray(ray.get(task))


def assemble_counts(
        rows: Iterable[np.ndarray],
        n_rows: int,
        n_cols: int,
        dtype: np.dtype,
        sparse: bool,
        desc: str,
) -> Dataset:
    """
    Build the (n_rows x n_cols) segment-count matrix from per-sample count rows.

    :param rows: iterable yielding one count row (array-like, length n_cols) per
        sample, in order.
    :param dtype: storage dtype for the counts (np.float32 or np.uint16).
    :param sparse: if True, return a scipy CSR matrix instead of a dense ndarray.
        SMILES/short sequences make the matrix >99% zeros, so CSR saves orders of
        magnitude. Note XGBoost treats CSR's unstored zeros as *missing* rather
        than explicit 0, so the same ``sparse`` value must be used at train and
        predict time.
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


class Loader:
    def __init__(
            self,
            file_label: Optional[FileLabel],
            folds: int = 0,
            fold_index: int = 0,
            count_dtype: np.dtype = np.float32,
            sparse: bool = False
    ) -> None:
        # LoaderInference passes None here and overrides the file-loading hooks.
        self.file_label: Optional[FileLabel] = file_label
        self.folds: int = folds
        self.fold_index: int = fold_index
        # Storage format for the segment-count matrix. See assemble_counts.
        self.count_dtype: np.dtype = count_dtype
        self.sparse: bool = sparse

        self.train_files: Optional[list[str]] = None
        self.test_files: Optional[list[str]] = None
        self.train_labels: Optional[list] = None
        self.test_labels: Optional[list] = None

        seq_manager.clear()
        self._load_sequence_files()
        self._get_train_seq()
        self._get_test_seq()

    def _load_sequence_files(self) -> None:
        # Only reached for the training loader; LoaderInference overrides this hook.
        assert self.file_label is not None
        self.train_files, self.test_files, self.train_labels, self.test_labels = self.file_label.get_train_test_path(
            num_folds=self.folds,
            fold_index=self.fold_index
        )

    @staticmethod
    @ray.remote
    def _get_one_sequence(file: str, alphabet: Alphabet) -> Sequence:
        # Ray workers are separate processes, so the alphabet has to travel with the task.
        set_alphabet(alphabet)
        return Sequence(file)

    def _get_train_seq(self):
        logger.info('Loading training sequences...')
        alphabet = ray.put(get_alphabet())
        train_sequences = [Loader._get_one_sequence.remote(file, alphabet) for file in self.train_files]
        train_sequences = [ray.get(a) for a in tqdm(train_sequences)]
        seq_manager.add_train_sequences(train_sequences)

    def _get_test_seq(self):
        if not self.test_files:
            return

        logger.info('Loading testing sequences...')
        alphabet = ray.put(get_alphabet())
        test_sequences = [Loader._get_one_sequence.remote(file, alphabet) for file in self.test_files]
        test_sequences = [ray.get(a) for a in tqdm(test_sequences)]
        seq_manager.add_test_sequences(test_sequences)

    @staticmethod
    @ray.remote
    def _get_one_kmer_dataset(seq: Sequence, k: int, alphabet: Alphabet) -> np.ndarray:
        """
        Deprecated. All kmers are now stored in the segment pool.
        """
        set_alphabet(alphabet)

        return seq.get_kmer_count(k)

    def get_kmer_dataset(self, k: int):
        """
        Deprecated. All kmers are now stored in the segment pool.
        """

        logger.info(f'Getting k-mer dataset for k={k}...')

        alphabet = ray.put(get_alphabet())
        train_kmer = [Loader._get_one_kmer_dataset.remote(seq, k, alphabet) for seq in seq_manager.train_sequences]
        test_kmer = [Loader._get_one_kmer_dataset.remote(seq, k, alphabet) for seq in seq_manager.test_sequences]

        return (
            np.asarray([ray.get(a) for a in tqdm(train_kmer)], dtype=np.float32),
            np.asarray([ray.get(b) for b in tqdm(test_kmer)], dtype=np.float32),
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )

    @staticmethod
    @ray.remote
    def _get_one_extended_dataset(seq: Sequence, seg_pool_: SegmentPool, alphabet: Alphabet) -> np.ndarray:
        # Ray workers are separate processes, so the alphabet has to travel with the task.
        set_alphabet(alphabet)
        return seq.get_count_from_seg_manager(seg_pool_)

    def get_dataset_from_pool(self):
        """
        Get the extended dataset for the training and test sequences
        :return: tuple: The training and test datasets
        """
        logger.info(f'Counting segments to generate the dataset...')

        # Combine training and testing sequences to maximise parallelism.
        all_sequences = seq_manager.train_sequences + seq_manager.test_sequences
        n_train = len(seq_manager.train_sequences)

        # Put the segment pool into the object store ONCE. Passing it straight to
        # .remote() would serialise a full copy per task (tens of thousands of
        # identical multi-MB objects), which floods the object store and spills to
        # disk. Ray dereferences these refs automatically, so the remote function
        # still receives the objects themselves.
        pool_ref = ray.put(seg_pool)
        alphabet_ref = ray.put(get_alphabet())

        tasks = [Loader._get_one_extended_dataset.remote(seq, pool_ref, alphabet_ref) for seq in all_sequences]

        # Stream results into the chosen representation (dense/sparse, float32/uint16).
        all_data = assemble_counts(
            _row_stream(tasks), len(all_sequences), len(seg_pool),
            dtype=self.count_dtype, sparse=self.sparse,
            desc='Counting segments for train/test'
        )

        # Separate the results back into training and testing datasets.
        train_data = all_data[:n_train]
        test_data = all_data[n_train:]

        return (
            train_data,
            test_data,
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )
