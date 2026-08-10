import json
import os
from typing import Optional

import pandas as pd
import numpy as np

from pgse.dataset.sample_source import SampleSource, Split
from pgse.log import logger


class FileLabel(SampleSource):
    """Samples held one per file, paired with their labels by a lookup table."""

    def __init__(
            self,
            label_file: str | dict,
            data_dir: Optional[str] = None,
            pre_kfold_info_file: Optional[str] = None
    ) -> None:
        """
        FileLabel constructor.
        :param label_file: Path to the CSV file containing the labels
        :param data_dir: Directory containing the data files When it is a dictionary, it should be in the format of
        {file1: label1, file2: label2, ...}
        """
        self.label_file: str | dict = label_file
        self.data_dir: Optional[str] = data_dir
        self.pre_kfold_info_file: Optional[str] = pre_kfold_info_file
        self.label_lookup: dict[str, str] = self._load_label_lookup()

        super().__init__(
            list(self.label_lookup.keys()),
            np.array(list(self.label_lookup.values()), dtype=np.float32)
        )

    def _load_label_lookup(self) -> dict[str, str]:
        if isinstance(self.label_file, str):
            data = pd.read_csv(self.label_file, dtype=str)
        elif isinstance(self.label_file, dict):
            data = pd.DataFrame(self.label_file.items(), columns=['files', 'labels'])
        else:
            raise ValueError('Invalid label file format')

        data['files'] = [
            p if os.path.exists(p) or not self.data_dir else os.path.join(self.data_dir, p)
            for p in data['files']
        ]

        # check if all files exist, remove those that do not exist
        kept = data[data['files'].apply(os.path.exists)]
        removed = data[~data['files'].apply(os.path.exists)]
        if len(removed) > 0:
            logger.warning(f'ignored data with missing files:\n{removed}')

        return kept.set_index('files').to_dict()['labels']

    def get_train_test_split(
            self,
            test_size: float = 0.2,
            random_state: int = 42,
            num_folds: int = 0,
            fold_index: int = 0
    ) -> Split:
        """Split the files into a training and a test set, honouring predefined folds.

        Args:
            test_size: Proportion of the files held out, read when num_folds is 0.
            random_state: Seed of the split.
            num_folds: Number of cross-validation folds. 0 takes a single random split.
            fold_index: Index of the fold held out for testing.
        """
        if not self.pre_kfold_info_file:
            return super().get_train_test_split(test_size, random_state, num_folds, fold_index)

        return self._pre_kfold_split(num_folds, fold_index)

    def _pre_kfold_split(self, num_folds: int, fold_index: int) -> Split:
        """Read the folds from the predefined k-fold file.

        Args:
            num_folds: Number of folds to read. 0 reads every fold in the file.
            fold_index: Index of the fold held out for testing.
        """
        if not self.data_dir:
            raise ValueError('pre_kfold_info_file needs data_dir, the directory holding the files it names.')

        data_dir = self.data_dir
        with open(str(self.pre_kfold_info_file), 'r') as f:
            k_fold_indices = json.load(f)

        # fold_index as the test set
        test_files = [os.path.join(data_dir, p) for p in k_fold_indices[f'fold_{fold_index}']]
        test_labels = [self.label_lookup[file] for file in test_files]

        # other folds as the training set
        train_files = []
        train_labels = []

        if num_folds > 0:
            for i in range(num_folds):
                if i != fold_index:
                    train_files.extend([os.path.join(data_dir, p) for p in k_fold_indices[f'fold_{i}']])
                    train_labels.extend([self.label_lookup[os.path.join(data_dir, file)] for file in k_fold_indices[f'fold_{i}']])
        else:
            # just load from the second fold till the end
            for i in range(1, len(k_fold_indices)):
                train_files.extend([os.path.join(data_dir, p) for p in k_fold_indices[f'fold_{i}']])
                train_labels.extend([self.label_lookup[os.path.join(data_dir, file)] for file in k_fold_indices[f'fold_{i}']])

        return train_files, test_files, np.array(train_labels, dtype=np.float32), np.array(test_labels, dtype=np.float32)
