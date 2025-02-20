import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from pgse.dataset.file_label import FileLabel


class TestFileLabel(unittest.TestCase):
    @patch('pgse.dataset.file_label.pd.read_csv')
    def setUp(self, mock_read_csv):
        mock_data = pd.DataFrame({
            'files': ['file1', 'file2', 'file3', 'file4'],
            'labels': ['0.1', '1.1', '0.1', '1.1']
        })
        mock_read_csv.return_value = mock_data
        self.file_label = FileLabel('dummy_label_file.csv', '/dummy/data/dir/')

    def test_train_test_path_splits_no_kfold(self):
        train_files, test_files, train_labels, test_labels = self.file_label.get_train_test_path()
        self.assertEqual(len(train_files), 3)
        self.assertEqual(len(test_files), 1)
        self.assertEqual(len(train_labels), 3)
        self.assertEqual(len(test_labels), 1)

    def test_train_test_path_kfold(self):
        train_files, test_files, train_labels, test_labels = self.file_label.get_train_test_path(num_folds=4, fold_index=0)
        self.assertEqual(len(train_files), 3)
        self.assertEqual(len(test_files), 1)
        self.assertEqual(len(train_labels), 3)
        self.assertEqual(len(test_labels), 1)
        train_files, test_files, train_labels, test_labels = self.file_label.get_train_test_path(num_folds=4, fold_index=2)
        self.assertEqual(len(train_files), 3)
        self.assertEqual(len(test_files), 1)
        self.assertEqual(len(train_labels), 3)
        self.assertEqual(len(test_labels), 1)

    def test_stratify_disabled(self):
        train_files, test_files, train_labels, test_labels = self.file_label.get_train_test_path(num_folds=2, fold_index=0)
        self.assertEqual(len(train_files), 3)
        self.assertEqual(len(test_files), 1)
        self.assertEqual(len(train_labels), 3)
        self.assertEqual(len(test_labels), 1)

