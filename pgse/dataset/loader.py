from typing import Optional

import numpy as np

from pgse.dataset.file_label import FileLabel
from tqdm import tqdm

from pgse.etc.alphabet import Alphabet, get_alphabet, set_alphabet
from pgse.genome import seq_manager
from pgse.log import logger
from pgse.genome.sequence import Sequence
import ray
from pgse.segment import seg_pool
from pgse.segment.segment_pool import SegmentPool


class Loader:
    def __init__(
            self,
            file_label: Optional[FileLabel],
            folds: int = 0,
            fold_index: int = 0
    ) -> None:
        # LoaderInference passes None here and overrides the file-loading hooks.
        self.file_label: Optional[FileLabel] = file_label
        self.folds: int = folds
        self.fold_index: int = fold_index

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

        # Put the segment pool into the object store ONCE. Passing it straight to
        # .remote() would serialise a full copy per task (tens of thousands of
        # identical multi-MB objects), which floods the object store and spills to
        # disk. Ray dereferences these refs automatically, so the remote function
        # still receives the objects themselves.
        pool_ref = ray.put(seg_pool)
        alphabet_ref = ray.put(get_alphabet())

        tasks = [Loader._get_one_extended_dataset.remote(seq, pool_ref, alphabet_ref) for seq in all_sequences]

        # Fetch the results for all tasks in parallel.
        all_data = np.asarray([ray.get(task) for task in tqdm(tasks, desc='Counting segments for train/test')], dtype=np.float32)

        # Separate the results back into training and testing datasets.
        train_data = all_data[:len(seq_manager.train_sequences)]
        test_data = all_data[len(seq_manager.train_sequences):]

        return (
            train_data,
            test_data,
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )
