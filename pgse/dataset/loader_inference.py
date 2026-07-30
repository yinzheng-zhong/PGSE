import numpy as np

from pgse.dataset.loader import Dataset, Loader
from pgse.genome import seq_manager
from pgse.log import logger


class LoaderInference(Loader):
    def __init__(
            self,
            files: list[str],
            count_dtype: np.dtype = np.float32,
            sparse: bool = False,
            workers: int = 8
    ):
        super().__init__(None, count_dtype=count_dtype, sparse=sparse, workers=workers)
        self.test_files = files

        self._get_test_seq()

    def _load_sequence_files(self):
        pass

    def _get_train_seq(self):
        pass

    def get_dataset_from_pool(self) -> Dataset:
        logger.info('Counting segments for test...')
        return self._count_dataset(seq_manager.test_sequences, 'Counting segments for test')
