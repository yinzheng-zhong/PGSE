import json
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from src.log import logger
from collections import Counter


class FileLabel:
    def __init__(self, label_file, data_dir, pre_kfold_info_file=None):
        self.label_file = label_file
        self.data_dir = data_dir
        self.pre_kfold_info_file = pre_kfold_info_file
        self.label_lookup = self._load_label_lookup()

    def _load_label_lookup(self):
        data = pd.read_csv(self.label_file, dtype=str)
        data['files'] = [os.path.join(self.data_dir, p) for p in data['files']]
        return data.set_index('files').to_dict()['labels']

    def _perform_train_test_split(self, files, labels, test_size, random_state):
        """
        :param files: List of filenames
        :param labels: Corresponding labels
        :param test_size: Test data proportion
        :param random_state: Random State
        :return: train_files, test_files, train_labels, test_labels
        """
        try:
            return train_test_split(
                files,
                labels,
                stratify=labels,
                test_size=test_size,
                random_state=random_state
            )
        except ValueError:
            logger.warning('Stratify disabled due to single instance class')
            return train_test_split(
                files,
                labels,
                test_size=test_size,
                random_state=random_state
            )

    def get_train_test_path(self, test_size=0.2, random_state=42, num_folds=0, fold_index=0):
        """

        :param test_size:
        :param random_state:
        :param num_folds:
        :param fold_index:
        :return:
        """
        files = list(self.label_lookup.keys())
        labels = np.array(list(self.label_lookup.values()), dtype=np.float32)

        if self.pre_kfold_info_file:
            with open(self.pre_kfold_info_file, 'r') as f:
                k_fold_indices = json.load(f)

            # fold_index as the test set
            test_files = [os.path.join(self.data_dir, p) for p in k_fold_indices[f'fold_{fold_index}']]
            test_labels = [self.label_lookup[file] for file in test_files]

            # other folds as the training set
            train_files = []
            train_labels = []

            if num_folds > 0:
                for i in range(num_folds):
                    if i != fold_index:
                        train_files.extend([os.path.join(self.data_dir, p) for p in k_fold_indices[f'fold_{i}']])
                        train_labels.extend([self.label_lookup[os.path.join(self.data_dir, file)] for file in k_fold_indices[f'fold_{i}']])
            else:
                # just load from the second fold till the end
                for i in range(1, len(k_fold_indices)):
                    train_files.extend([os.path.join(self.data_dir, p) for p in k_fold_indices[f'fold_{i}']])
                    train_labels.extend([self.label_lookup[os.path.join(self.data_dir, file)] for file in k_fold_indices[f'fold_{i}']])

            return train_files, test_files, np.array(train_labels, dtype=np.float32), np.array(test_labels, dtype=np.float32)

        if num_folds <= 0:
            return self._perform_train_test_split(files, labels.astype(np.int32), test_size, random_state)
        else:
            try:
                k_fold_instance = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
                splits = list(k_fold_instance.split(files, labels.astype(np.int32)))
            except ValueError as e:
                logger.warning(f'StratifiedKFold failed: {e}. Falling back to KFold.')
                k_fold_instance = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
                splits = list(k_fold_instance.split(files))

            train_index, test_index = splits[fold_index]

            train_files = [files[i] for i in train_index]
            test_files = [files[i] for i in test_index]
            train_labels = labels[train_index]
            test_labels = labels[test_index]

            return train_files, test_files, train_labels, test_labels
