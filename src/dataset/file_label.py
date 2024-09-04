import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.log import logger
from collections import Counter


class FileLabel:
    def __init__(self, label_file, data_dir):
        self.label_file = label_file
        self.data_dir = data_dir
        self.label_lookup = self._load_label_lookup()

    def _load_label_lookup(self):
        data = pd.read_csv(self.label_file, dtype=str)
        data['files'] = self.data_dir + data['files']
        return data.set_index('files').to_dict()['labels']

    def get_train_test_path(self, test_size=0.2, random_state=42):
        """
        To get the path and labels, so they can be loaded later
        :param test_size:
        :param random_state:
        :return:
        """
        files = list(self.label_lookup.keys())
        labels = np.array(list(self.label_lookup.values()), dtype=np.float32)

        # Ensure there's more than one instance of each class, if not, stratify=None
        counts = np.array(list(Counter(labels).values()))
        min_class_size = min(counts)

        if min_class_size < 2:
            logger.warning('Stratify is disabled because there is at least one class with only one instance')
            stratify = None
        else:
            stratify = labels

        train_files, test_files, train_labels, test_labels = train_test_split(
            files,
            labels,
            stratify=stratify,
            test_size=test_size,
            random_state=random_state
        )

        return train_files, test_files, train_labels, test_labels
