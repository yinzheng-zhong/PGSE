import numpy as np
import ray
from tqdm import tqdm

from pgse.dataset.loader import Loader
from pgse.genome import seq_manager
from pgse.log import logger
from pgse.segment import seg_pool


class LoaderInference(Loader):
    def __init__(self, files: [str]):
        super().__init__(None)
        self.test_files = files

        self._get_test_seq()

    def _load_sequence_files(self):
        pass

    def _get_train_seq(self):
        pass

    def get_dataset_from_pool(self):
        logger.info('Counting segments for test...')
        tasks = [Loader._get_one_extended_dataset.remote(seq, seg_pool) for seq in seq_manager.test_sequences]
        data = np.asarray([ray.get(task) for task in tqdm(tasks, desc='Counting segments for train/test')], dtype=np.float32)

        return data