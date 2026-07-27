import numpy as np
import ray
from tqdm import tqdm

from pgse.dataset.loader import Loader
from pgse.etc.alphabet import get_alphabet
from pgse.genome import seq_manager
from pgse.log import logger
from pgse.segment import seg_pool


class LoaderInference(Loader):
    def __init__(self, files: list[str]):
        super().__init__(None)
        self.test_files = files

        self._get_test_seq()

    def _load_sequence_files(self):
        pass

    def _get_train_seq(self):
        pass

    def get_dataset_from_pool(self) -> np.ndarray:
        logger.info('Counting segments for test...')
        # One shared copy of the pool in the object store, not one per task. See
        # Loader.get_dataset_from_pool.
        pool_ref = ray.put(seg_pool)
        alphabet_ref = ray.put(get_alphabet())
        tasks = [Loader._get_one_extended_dataset.remote(seq, pool_ref, alphabet_ref) for seq in seq_manager.test_sequences]
        data = np.asarray([ray.get(task) for task in tqdm(tasks, desc='Counting segments for train/test')], dtype=np.float32)

        return data