from abc import ABC
from typing import Optional, Sequence

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from pgse.log import logger

Split = tuple[list[str], list[str], np.ndarray, np.ndarray]


class SampleSource(ABC):
    """The samples of a dataset and the way they are split into training and test sets."""

    # True when every item is the sequence itself rather than a path to read it from.
    inline: bool = False

    def __init__(self, items: list[str], labels: np.ndarray, label_names: Sequence[str]) -> None:
        """
        Args:
            items: One entry per sample: a path to a sequence file, or the sequence
                text itself when the source is inline.
            labels: One row per sample and one column per label, in the order of items.
            label_names: Name of each label, in the order of the label columns.

        Raises:
            ValueError: If the labels do not hold one column per name.
        """
        self.items: list[str] = items
        self.label_names: list[str] = [str(name) for name in label_names]

        given = np.asarray(labels, dtype=np.float32)
        if given.size != len(items) * len(self.label_names):
            raise ValueError(
                f'Got {given.size} labels for {len(items)} samples carrying '
                f'{len(self.label_names)} labels each.'
            )

        self.labels: np.ndarray = given.reshape(len(items), len(self.label_names))

    def __len__(self) -> int:
        return len(self.items)

    @property
    def n_labels(self) -> int:
        """How many labels every sample carries."""
        return len(self.label_names)

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

    def _stratify_labels(self) -> Optional[np.ndarray]:
        """A discrete copy of the labels to stratify on, or None with several labels."""
        if self.n_labels != 1:
            logger.info(f'Stratification is off: the samples carry {self.n_labels} labels.')
            return None

        return self.labels[:, 0].astype(np.int32)

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
                stratify=self._stratify_labels(),
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
        train_index, test_index = self._fold_indices(num_folds, fold_index, random_state)

        return (
            [self.items[i] for i in train_index],
            [self.items[i] for i in test_index],
            self.labels[train_index],
            self.labels[test_index]
        )

    def _fold_indices(self, num_folds: int, fold_index: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
        """The training and test positions of one fold, stratified where that is possible.

        Args:
            num_folds: Number of folds.
            fold_index: Index of the fold held out for testing.
            random_state: Seed of the split.
        """
        stratify = self._stratify_labels()

        if stratify is not None:
            try:
                k_fold_instance = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
                return list(k_fold_instance.split(self.items, stratify))[fold_index]
            except ValueError as e:
                logger.warning(f'StratifiedKFold failed: {e}. Falling back to KFold.')

        k_fold_instance = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        return list(k_fold_instance.split(self.items))[fold_index]
