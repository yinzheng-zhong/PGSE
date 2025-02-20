import numpy as np

from pgse.dataset.file_label import FileLabel
from tqdm import tqdm

from pgse.genome import seq_manager
from pgse.log import logger
from pgse.genome.sequence import Sequence
import ray
from pgse.segment import seg_pool


class Loader:
    def __init__(
            self,
            file_label: FileLabel,
            folds: int = 0,
            fold_index: int = 0
    ):
        self.file_label = file_label
        self.folds = folds
        self.fold_index = fold_index

        self.train_files = None
        self.test_files = None
        self.train_labels = None
        self.test_labels = None

        seq_manager.clear()
        self._load_sequence_files()
        self._get_train_seq()
        self._get_test_seq()

    def _load_sequence_files(self):
        self.train_files, self.test_files, self.train_labels, self.test_labels = self.file_label.get_train_test_path(
            num_folds=self.folds,
            fold_index=self.fold_index
        )

    @staticmethod
    @ray.remote
    def _get_one_sequence(file):
        return Sequence(file)

    def _get_train_seq(self):
        logger.info('Loading training sequences...')
        train_sequences = [Loader._get_one_sequence.remote(file) for file in self.train_files]
        train_sequences = [ray.get(a) for a in tqdm(train_sequences)]
        seq_manager.add_train_sequences(train_sequences)

    def _get_test_seq(self):
        if not self.test_files:
            return

        logger.info('Loading testing sequences...')
        test_sequences = [Loader._get_one_sequence.remote(file) for file in self.test_files]
        test_sequences = [ray.get(a) for a in tqdm(test_sequences)]
        seq_manager.add_test_sequences(test_sequences)

    @staticmethod
    @ray.remote
    def _get_one_kmer_dataset(seq, k):
        """
        Deprecated. All kmers are now stored in the segment pool.
        """

        return seq.get_kmer_count(k)

    def get_kmer_dataset(self, k: int):
        """
        Deprecated. All kmers are now stored in the segment pool.
        """

        logger.info(f'Getting k-mer dataset for k={k}...')

        train_kmer = [Loader._get_one_kmer_dataset.remote(seq, k) for seq in seq_manager.train_sequences]
        test_kmer = [Loader._get_one_kmer_dataset.remote(seq, k) for seq in seq_manager.test_sequences]

        return (
            np.asarray([ray.get(a) for a in tqdm(train_kmer)], dtype=np.float32),
            np.asarray([ray.get(b) for b in tqdm(test_kmer)], dtype=np.float32),
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )

    @staticmethod
    @ray.remote
    def _get_one_extended_dataset(seq, seg_pool_):
        return seq.get_count_from_seg_manager(seg_pool_)

    def get_dataset_from_pool(self):
        """
        Get the extended dataset for the training and test sequences
        :return: tuple: The training and test datasets
        """
        logger.info(f'Counting segments to generate the dataset...')

        # Combine training and testing sequences to maximise parallelism.
        all_sequences = seq_manager.train_sequences + seq_manager.test_sequences
        tasks = [Loader._get_one_extended_dataset.remote(seq, seg_pool) for seq in all_sequences]

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
