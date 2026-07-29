import numpy as np
import ray

from pgse.dataset.loader import Dataset, Loader, _row_stream, assemble_counts
from pgse.etc.alphabet import get_alphabet
from pgse.genome import seq_manager
from pgse.log import logger
from pgse.segment import seg_pool


class LoaderInference(Loader):
    def __init__(self, files: list[str], count_dtype: np.dtype = np.float32, sparse: bool = False):
        super().__init__(None, count_dtype=count_dtype, sparse=sparse)
        self.test_files = files

        self._get_test_seq()

    def _load_sequence_files(self):
        pass

    def _get_train_seq(self):
        pass

    def get_dataset_from_pool(self) -> Dataset:
        logger.info('Counting segments for test...')
        # One shared copy of the pool in the object store, not one per task. See
        # Loader.get_dataset_from_pool.
        pool_ref = ray.put(seg_pool)
        alphabet_ref = ray.put(get_alphabet())
        tasks = [Loader._get_one_extended_dataset.remote(seq, pool_ref, alphabet_ref) for seq in seq_manager.test_sequences]

        return assemble_counts(
            _row_stream(tasks), len(seq_manager.test_sequences), len(seg_pool),
            dtype=self.count_dtype, sparse=self.sparse,
            desc='Counting segments for test'
        )
