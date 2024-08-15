import numpy as np

from src.dataset.file_label import FileLabel
from tqdm import tqdm

from src.genome import seq_manager
from src.log import logger
from src.genome.sequence import Sequence
import ray
from src.segment import seg_manager


class Loader:
    def __init__(
            self,
            file_label: FileLabel,
    ):
        self.file_label = file_label

        self.train_files = None
        self.test_files = None
        self.train_labels = None
        self.test_labels = None

        self._load_sequence_files()
        self._get_train_seq()
        self._get_test_seq()

    def _load_sequence_files(self):
        self.train_files, self.test_files, self.train_labels, self.test_labels = self.file_label.get_train_test_path()

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
        logger.info('Loading test sequences...')
        test_sequences = [Loader._get_one_sequence.remote(file) for file in self.test_files]
        test_sequences = [ray.get(a) for a in tqdm(test_sequences)]
        seq_manager.add_test_sequences(test_sequences)

    @staticmethod
    @ray.remote
    def _get_one_kmer_dataset(args):
        seq, k = args
        return seq.get_kmer_count(k)

    def get_kmer_dataset(self, k: int):

        logger.info(f'Getting k-mer dataset for k={k}...')

        train_kmer = [Loader._get_one_kmer_dataset.remote((seq, k)) for seq in seq_manager.train_sequences]
        test_kmer = [Loader._get_one_kmer_dataset.remote((seq, k)) for seq in seq_manager.test_sequences]

        return (
            np.asarray([ray.get(a) for a in tqdm(train_kmer)], dtype=np.float32),
            np.asarray([ray.get(b) for b in tqdm(test_kmer)], dtype=np.float32),
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )

    @staticmethod
    @ray.remote
    def _get_one_extended_dataset(seq):
        return seq.get_count_from_seg_manager()

    def get_extended_dataset(self):
        """
        Get the extended dataset for the training and test sequences
        :return: tuple: The training and test datasets
        """
        logger.info(f'Getting extended dataset...')
        train_ext = [Loader._get_one_extended_dataset.remote(seq, seg_manager) for seq in seq_manager.train_sequences]
        test_ext = [Loader._get_one_extended_dataset.remote(seq, seg_manager) for seq in seq_manager.test_sequences]


        return (
            np.asarray([ray.get(a) for a in tqdm(train_ext)], dtype=np.float32),
            np.asarray([ray.get(b) for b in tqdm(test_ext)], dtype=np.float32),
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )
