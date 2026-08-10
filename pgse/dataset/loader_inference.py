import numpy as np
import numpy.typing as npt

from pgse.dataset.loader import Dataset, Loader
from pgse.genome import seq_manager
from pgse.log import logger


class LoaderInference(Loader):
    def __init__(
            self,
            items: list[str],
            inline: bool = False,
            count_dtype: npt.DTypeLike = np.float32,
            sparse: bool = False,
            workers: int = 8
    ):
        """
        Args:
            items: The samples to score: file paths, or the sequences themselves when
                inline is set.
            inline: The items are sequences held in memory rather than paths to read.
            count_dtype: Storage dtype of the count matrix (np.float32 or np.uint16).
            sparse: Store the count matrix as a sparse CSR matrix.
            workers: Threads used for counting.
        """
        super().__init__(None, count_dtype=count_dtype, sparse=sparse, workers=workers)
        self.inline = inline
        self.test_items = items

        self._get_test_seq()

    def _load_sequence_files(self):
        pass

    def _get_train_seq(self):
        pass

    def get_dataset_from_pool(self) -> Dataset:
        logger.info('Counting segments for test...')
        return self._count_dataset(seq_manager.test_sequences, 'Counting segments for test')
