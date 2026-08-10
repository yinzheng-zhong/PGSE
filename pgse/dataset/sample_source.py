from abc import ABC

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from pgse.log import logger

Split = tuple[list[str], list[str], np.ndarray, np.ndarray]


class SampleSource(ABC):
    """The samples of a dataset and the way they are split into training and test sets."""

    # True when every item is the sequence itself rather than a path to read it from.
    inline: bool = False

    def __init__(self, items: list[str], labels: np.ndarray) -> None:
        """
        Args:
            items: One entry per sample: a path to a sequence file, or the sequence
                text itself when the source is inline.
            labels: The target value of each sample, in the same order as items.
        """
        self.items: list[str] = items
        self.labels: np.ndarray = labels

    def __len__(self) -> int:
        return len(self.items)

    def get_train_test_split(
            self,
            test_size: float = 0.2,
            random_state: int = 42,
            num_folds: int = 0,
            fold_index: int = 0
    ) -> Split:
        """Split the samples into a training and a test set.

        Args:
            test_size: Proportion of the samples held out, read when num_folds is 0.
            random_state: Seed of the split.
            num_folds: Number of cross-validation folds. 0 takes a single random split.
            fold_index: Index of the fold held out for testing.
        """
        if num_folds <= 0:
            return self._random_split(test_size, random_state)

        return self._fold_split(num_folds, fold_index, random_state)

    def _random_split(self, test_size: float, random_state: int) -> Split:
        """Hold out a proportion of the samples, stratified on a discrete copy of the labels.

        Args:
            test_size: Proportion of the samples held out.
            random_state: Seed of the split.
        """
        try:
            return train_test_split(
                self.items,
                self.labels,
                stratify=self.labels.astype(np.int32),
                test_size=test_size,
                random_state=random_state
            )
        except ValueError:
            logger.warning('Stratify disabled due to single instance class')
            return train_test_split(
                self.items,
                self.labels,
                test_size=test_size,
                random_state=random_state
            )

    def _fold_split(self, num_folds: int, fold_index: int, random_state: int) -> Split:
        """Hold out one fold of a k-fold split.

        Args:
            num_folds: Number of folds.
            fold_index: Index of the fold held out for testing.
            random_state: Seed of the split.
        """
        try:
            k_fold_instance = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
            splits = list(k_fold_instance.split(self.items, self.labels.astype(np.int32)))
        except ValueError as e:
            logger.warning(f'StratifiedKFold failed: {e}. Falling back to KFold.')
            k_fold_instance = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
            splits = list(k_fold_instance.split(self.items))

        train_index, test_index = splits[fold_index]

        return (
            [self.items[i] for i in train_index],
            [self.items[i] for i in test_index],
            self.labels[train_index],
            self.labels[test_index]
        )
