import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from pgse.dataset.file_label import FileLabel


class TestFileLabel(unittest.TestCase):
    @patch('pgse.dataset.file_label.os.path.exists', new=lambda path: True)
    @patch('pgse.dataset.file_label.pd.read_csv')
    def setUp(self, mock_read_csv):
        mock_data = pd.DataFrame({
            'files': ['file1', 'file2', 'file3', 'file4'],
            'labels': ['0.1', '1.1', '0.1', '1.1']
        })
        mock_read_csv.return_value = mock_data
        self.file_label = FileLabel('dummy_label_file.csv', '/dummy/data/dir/', label_columns='labels')

    def test_train_test_path_splits_no_kfold(self):
        train_files, test_files, train_labels, test_labels = self.file_label.get_train_test_split()
        self.assertEqual(len(train_files), 3)
        self.assertEqual(len(test_files), 1)
        self.assertEqual(len(train_labels), 3)
        self.assertEqual(len(test_labels), 1)

    def test_train_test_path_kfold(self):
        train_files, test_files, train_labels, test_labels = self.file_label.get_train_test_split(num_folds=4, fold_index=0)
        self.assertEqual(len(train_files), 3)
        self.assertEqual(len(test_files), 1)
        self.assertEqual(len(train_labels), 3)
        self.assertEqual(len(test_labels), 1)
        train_files, test_files, train_labels, test_labels = self.file_label.get_train_test_split(num_folds=4, fold_index=2)
        self.assertEqual(len(train_files), 3)
        self.assertEqual(len(test_files), 1)
        self.assertEqual(len(train_labels), 3)
        self.assertEqual(len(test_labels), 1)

    def test_two_fold_split(self):
        train_files, test_files, train_labels, test_labels = self.file_label.get_train_test_split(num_folds=2, fold_index=0)
        self.assertEqual(len(train_files), 2)
        self.assertEqual(len(test_files), 2)
        self.assertEqual(len(train_labels), 2)
        self.assertEqual(len(test_labels), 2)

    def test_files_are_not_inline(self):
        self.assertFalse(self.file_label.inline)

    def test_one_label_is_a_single_column(self):
        self.assertEqual(['labels'], self.file_label.label_names)
        self.assertEqual((4, 1), self.file_label.labels.shape)
        np.testing.assert_allclose([0.1, 1.1, 0.1, 1.1], self.file_label.labels[:, 0], rtol=1e-6)


class TestFileLabelColumns(unittest.TestCase):
    """The sample files always come from the files column, the labels from the named ones."""

    def source(self, table: pd.DataFrame, label_columns) -> FileLabel:
        """Build a source over a label table whose files all exist.

        Args:
            table: The label table to read.
            label_columns: The label columns to name.
        """
        with patch('pgse.dataset.file_label.os.path.exists', new=lambda path: True), \
                patch('pgse.dataset.file_label.pd.read_csv', return_value=table):
            return FileLabel('labels.csv', '/data/', label_columns=label_columns)

    def test_several_label_columns_become_several_outputs(self):
        source = self.source(
            pd.DataFrame({'files': ['a.fna', 'b.fna'], 'mic': [2.0, 8.0], 'growth': [0.5, 1.5]}),
            ['mic', 'growth']
        )

        self.assertEqual(['mic', 'growth'], source.label_names)
        self.assertEqual(2, source.n_labels)
        np.testing.assert_allclose([[2.0, 0.5], [8.0, 1.5]], source.labels)

    def test_the_label_columns_are_read_in_the_order_given(self):
        source = self.source(
            pd.DataFrame({'files': ['a.fna'], 'mic': [2.0], 'growth': [0.5]}),
            ['growth', 'mic']
        )

        self.assertEqual(['growth', 'mic'], source.label_names)
        np.testing.assert_allclose([[0.5, 2.0]], source.labels)

    def test_unnamed_label_columns_are_rejected(self):
        with self.assertRaises(ValueError) as caught:
            FileLabel('labels.csv', '/data/')
        self.assertIn('files', str(caught.exception))

    def test_a_missing_label_column_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.source(pd.DataFrame({'files': ['a.fna'], 'mic': [2.0]}), ['mic', 'growth'])
        self.assertIn('growth', str(caught.exception))

    def test_a_missing_files_column_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.source(pd.DataFrame({'genome': ['a.fna'], 'mic': [2.0]}), 'mic')
        self.assertIn('files', str(caught.exception))

    def test_a_dict_of_labels_names_its_column(self):
        with patch('pgse.dataset.file_label.os.path.exists', new=lambda path: True):
            source = FileLabel({'a.fna': 2.0, 'b.fna': 8.0}, '/data/', label_columns='mic')

        self.assertEqual(['mic'], source.label_names)
        np.testing.assert_allclose([[2.0], [8.0]], source.labels)

    def test_a_dict_can_hold_several_labels_per_file(self):
        with patch('pgse.dataset.file_label.os.path.exists', new=lambda path: True):
            source = FileLabel(
                {'a.fna': {'mic': 2.0, 'growth': 0.5}, 'b.fna': {'mic': 8.0, 'growth': 1.5}},
                '/data/',
                label_columns=['mic', 'growth']
            )

        np.testing.assert_allclose([[2.0, 0.5], [8.0, 1.5]], source.labels)

    def test_a_dict_of_single_labels_needs_a_single_column(self):
        with patch('pgse.dataset.file_label.os.path.exists', new=lambda path: True):
            with self.assertRaises(ValueError) as caught:
                FileLabel({'a.fna': 2.0}, '/data/', label_columns=['mic', 'growth'])

        self.assertIn('mapping', str(caught.exception))
