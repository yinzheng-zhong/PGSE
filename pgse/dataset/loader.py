from typing import Iterator, Optional

import numpy as np
import numpy.typing as npt

from pgse.dataset.file_label import FileLabel
from tqdm import tqdm

from pgse.dataset.alphabet import Alphabet, get_alphabet, set_alphabet
from pgse.dataset.counts import Dataset, assemble_counts
from pgse.genome import seq_manager
from pgse.log import logger
from pgse.genome.sequence import Sequence
import ray
from pgse.algos import native_counter
from pgse.segment import seg_pool
from pgse.segment.segment_pool import SegmentPool


def _row_stream(tasks: list) -> Iterator[np.ndarray]:
    """
    Yield each Ray task's count row one at a time so the finished object can be
    released before the next is fetched, instead of holding all rows at once.
    """
    for task in tasks:
        yield np.asarray(ray.get(task))


class Loader:
    def __init__(
            self,
            file_label: Optional[FileLabel],
            folds: int = 0,
            fold_index: int = 0,
            count_dtype: npt.DTypeLike = np.float32,
            sparse: bool = False,
            workers: int = 8,
            dist: bool = False,
            nodes: int = 1
    ) -> None:
        # LoaderInference passes None here and overrides the file-loading hooks.
        self.file_label: Optional[FileLabel] = file_label
        self.folds: int = folds
        self.fold_index: int = fold_index
        # Storage format for the segment-count matrix. See assemble_counts.
        self.count_dtype: npt.DTypeLike = count_dtype
        self.sparse: bool = sparse
        # Parallelism for the native counting kernel: rayon threads per node, and how
        # many nodes to shard across when running distributed.
        self.workers: int = workers
        self.dist: bool = dist
        self.nodes: int = nodes

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

    @staticmethod
    @ray.remote
    def _count_shard_native(norm_segments, complement, node_lists, threads, alphabet) -> tuple:
        # One shard per node: build the shared automaton once and count this shard's
        # sequences against it. Returns a CSR triplet (indptr, indices, data).
        set_alphabet(alphabet)
        matcher = native_counter.raw_matcher(norm_segments, complement)
        return matcher.count_csr(node_lists, threads)

    def get_dataset_from_pool(self):
        """
        Get the extended dataset for the training and test sequences
        :return: tuple: The training and test datasets
        """
        logger.info(f'Counting segments to generate the dataset...')

        # Combine training and testing sequences to maximise parallelism.
        all_sequences = seq_manager.train_sequences + seq_manager.test_sequences
        n_train = len(seq_manager.train_sequences)

        all_data = self._count_dataset(all_sequences, 'Counting segments for train/test')

        # Separate the results back into training and testing datasets.
        train_data = all_data[:n_train]
        test_data = all_data[n_train:]

        return (
            train_data,
            test_data,
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )

    def _count_dataset(self, sequences, desc: str) -> Dataset:
        """
        Count the segment pool against every sequence, returning the count matrix in
        the configured representation (dense/sparse, float32/uint16).

        Uses the native Rust kernel when available: the automaton is built once and
        shared across rayon threads (single node) or once per node (distributed).
        Falls back to the original Ray-per-sequence C/Python counter otherwise.
        
        @param sequences: list[Sequence]: The sequences to count against.
        @param desc: str: Description for the progress bar.
        """
        n_cols = len(seg_pool)
        threads = max(self.workers, 1)

        if native_counter.native_available():
            if self.dist and self.nodes > 1:
                return self._count_native_dist(sequences, n_cols, threads)
            matcher = native_counter.build_matcher(seg_pool.get_copy())
            return native_counter.count_matrix(
                matcher, sequences, n_cols,
                dtype=self.count_dtype, sparse=self.sparse, threads=threads, desc=desc
            )

        return self._count_dataset_fallback(sequences, n_cols, desc)

    def _count_native_dist(self, sequences, n_cols: int, threads: int) -> Dataset:
        """Distributed counting: shard the sequences across nodes, one automaton each."""
        alphabet = get_alphabet()
        norm_segments = [alphabet.normalise(s) for s in seg_pool.get_copy()]
        complement = native_counter.complement_bytes(alphabet)

        # Share the (large) segment list once instead of per shard.
        seg_ref = ray.put(norm_segments)
        comp_ref = ray.put(complement)
        alph_ref = ray.put(alphabet)

        shards = np.array_split(np.arange(len(sequences)), max(self.nodes, 1))
        tasks = []
        for shard in shards:
            node_lists = [native_counter.normalised_nodes(sequences[int(i)], alphabet) for i in shard]
            tasks.append(
                Loader._count_shard_native.options(num_cpus=threads).remote(
                    seg_ref, comp_ref, node_lists, threads, alph_ref
                )
            )

        triplets = [ray.get(t) for t in tqdm(tasks, desc='Counting segments (distributed)')]
        return native_counter.assemble_from_triplets(
            triplets, len(sequences), n_cols, dtype=self.count_dtype, sparse=self.sparse
        )

    def _count_dataset_fallback(self, sequences, n_cols: int, desc: str) -> Dataset:
        """Original path: one Ray task per sequence, rebuilding the automaton each time."""
        # Put the segment pool into the object store ONCE. Passing it straight to
        # .remote() would serialise a full copy per task (tens of thousands of
        # identical multi-MB objects), which floods the object store and spills to
        # disk. Ray dereferences these refs automatically, so the remote function
        # still receives the objects themselves.
        pool_ref = ray.put(seg_pool)
        alphabet_ref = ray.put(get_alphabet())

        tasks = [Loader._get_one_extended_dataset.remote(seq, pool_ref, alphabet_ref) for seq in sequences]

        # Stream results into the chosen representation (dense/sparse, float32/uint16).
        return assemble_counts(
            _row_stream(tasks), len(sequences), n_cols,
            dtype=self.count_dtype, sparse=self.sparse,
            desc=desc
        )
