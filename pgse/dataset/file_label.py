import json
import os
from typing import Mapping, Optional

import pandas as pd
import numpy as np

from pgse.dataset.label_utils import LabelColumns, as_label_columns, missing_columns, to_float_matrix
from pgse.dataset.sample_source import SampleSource, Split
from pgse.log import logger

# The column of the label file naming the sequence file of each sample.
FILE_COLUMN = 'files'


class FileLabel(SampleSource):
    """Samples held one per file, paired with their labels by a lookup table."""

    def __init__(
            self,
            label_file: str | dict,
            data_dir: Optional[str] = None,
            pre_kfold_info_file: Optional[str] = None,
            label_columns: Optional[LabelColumns] = None
    ) -> None:
        """
        Args:
            label_file: Path of the CSV file pairing each sample file with its labels, or
                a dict in its place: {file: label} with one label column, and
                {file: {name: label}} with several.
            data_dir: Directory holding the sample files. Names in the label file are
                resolved against it.
            pre_kfold_info_file: Path of the JSON file holding predefined folds.
            label_columns: Name of the column holding the target value of each sample, or
                the names of several such columns to train one output per column. The
                sample files are always read from the 'files' column.

        Raises:
            ValueError: If no label column is named.
        """
        if label_columns is None:
            raise ValueError(
                f'Name the label column(s) of the label file, e.g. label_columns=["mic"]. '
                f'Its sample files are always read from the {FILE_COLUMN!r} column.'
            )

        self.label_file: str | dict = label_file
        self.data_dir: Optional[str] = data_dir
        self.pre_kfold_info_file: Optional[str] = pre_kfold_info_file
        self.label_columns: list[str] = as_label_columns(label_columns)
        self.label_lookup: dict[str, np.ndarray] = self._load_label_lookup()

        super().__init__(
            list(self.label_lookup.keys()),
            np.asarray(list(self.label_lookup.values()), dtype=np.float32),
            self.label_columns
        )

    def _load_label_lookup(self) -> dict[str, np.ndarray]:
        """Pair the path of every sample file that exists with its labels."""
        data = self._read_label_table()

        missing = missing_columns(data, [FILE_COLUMN] + self.label_columns)
        if missing:
            raise ValueError(
                f'The label file has no column {missing} to read. Its columns are {list(data.columns)}.'
            )

        data[FILE_COLUMN] = [
            p if os.path.exists(p) or not self.data_dir else os.path.join(self.data_dir, p)
            for p in data[FILE_COLUMN].astype(str)
        ]

        # check if all files exist, remove those that do not exist
        kept = data[data[FILE_COLUMN].apply(os.path.exists)]
        removed = data[~data[FILE_COLUMN].apply(os.path.exists)]
        if len(removed) > 0:
            logger.warning(f'ignored data with missing files:\n{removed}')

        labels = to_float_matrix(kept, self.label_columns)
        return {path: row for path, row in zip(kept[FILE_COLUMN], labels)}

    def _read_label_table(self) -> pd.DataFrame:
        """Read the label file, or build the same table from the dict given in its place."""
        if isinstance(self.label_file, str):
            return pd.read_csv(self.label_file)
        if isinstance(self.label_file, dict):
            return self._table_from_dict(self.label_file)

        raise ValueError('Invalid label file format')

    def _table_from_dict(self, labels: dict) -> pd.DataFrame:
        """Turn {file: label}, or {file: {name: label}}, into a table with a file column.

        Args:
            labels: The labels of each sample file.
        """
        rows = []
        for path, value in labels.items():
            if isinstance(value, Mapping):
                rows.append({FILE_COLUMN: path, **value})
            elif len(self.label_columns) == 1:
                rows.append({FILE_COLUMN: path, self.label_columns[0]: value})
            else:
                raise ValueError(
                    f'{path!r} carries a single label, but {self.label_columns} were asked for. '
                    f'Give each file a mapping of label name to value.'
                )

        return pd.DataFrame(rows)

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

        return (
            train_files,
            test_files,
            np.asarray(train_labels, dtype=np.float32).reshape(len(train_files), -1),
            np.asarray(test_labels, dtype=np.float32).reshape(len(test_files), -1)
        )
