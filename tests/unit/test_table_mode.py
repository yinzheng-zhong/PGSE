import tempfile
import unittest

import numpy as np
import pandas as pd

from pgse.dataset.alphabet import reset_alphabet, set_alphabet
from pgse.dataset.loader import Loader
from pgse.dataset.source_factory import build_source
from pgse.dataset.table_label import TableLabel
from pgse.genome import seq_manager

SMILES = ['CCO', 'CCN', 'CCC', 'CCCl']


def table(**columns) -> pd.DataFrame:
    """Build a table with the given columns.

    Args:
        columns: One entry per column, mapping its name to its values.
    """
    return pd.DataFrame(columns)


class TestTableLabel(unittest.TestCase):
    def setUp(self):
        self.source = TableLabel(
            table(smiles=SMILES, active=[True, False, True, False]),
            'smiles',
            'active'
        )

    def test_reads_the_named_columns(self):
        self.assertEqual(self.source.items, SMILES)
        np.testing.assert_array_equal(self.source.labels[:, 0], np.array([1, 0, 1, 0], dtype=np.float32))
        self.assertEqual(self.source.label_names, ['active'])
        self.assertEqual(len(self.source), 4)

    def test_samples_are_inline(self):
        self.assertTrue(self.source.inline)

    def test_reads_a_csv_file(self):
        path = self.write_csv()
        source = TableLabel(path, 'smiles', 'value')
        self.assertEqual(source.items, SMILES)
        np.testing.assert_array_equal(source.labels[:, 0], np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32))

    def test_numeric_strings_become_labels(self):
        source = TableLabel(table(text=['ab', 'cd'], labels=['0.25', '4']), 'text')
        np.testing.assert_array_equal(source.labels[:, 0], np.array([0.25, 4.0], dtype=np.float32))

    def test_empty_rows_are_dropped(self):
        source = TableLabel(
            table(text=['ab', None, '  ', 'cd'], labels=[1.0, 2.0, 3.0, None]),
            'text'
        )
        self.assertEqual(source.items, ['ab'])
        np.testing.assert_array_equal(source.labels[:, 0], np.array([1.0], dtype=np.float32))

    def test_unknown_column_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            TableLabel(table(smiles=SMILES, active=[1, 0, 1, 0]), 'sequence', 'active')
        self.assertIn('sequence', str(caught.exception))

    def test_non_numeric_labels_are_rejected(self):
        with self.assertRaises(ValueError) as caught:
            TableLabel(table(text=['ab', 'cd'], labels=['high', 'low']), 'text')
        self.assertIn('high', str(caught.exception))

    def test_split_without_folds(self):
        train, test, train_labels, test_labels = self.source.get_train_test_split(test_size=0.5)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)
        self.assertEqual(len(train_labels), 2)
        self.assertEqual(len(test_labels), 2)
        self.assertEqual(sorted(train + test), sorted(SMILES))

    def test_split_with_folds(self):
        seen = []
        for fold_index in range(4):
            train, test, _, _ = self.source.get_train_test_split(num_folds=4, fold_index=fold_index)
            self.assertEqual(len(train), 3)
            self.assertEqual(len(test), 1)
            seen.extend(test)

        self.assertEqual(sorted(seen), sorted(SMILES))

    def test_split_falls_back_when_stratifying_fails(self):
        source = TableLabel(table(text=['ab', 'cd', 'ef', 'gh'], labels=[0, 0, 0, 1]), 'text')
        train, test, _, _ = source.get_train_test_split(num_folds=4, fold_index=0)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 1)

    def write_csv(self) -> str:
        """Write a table to a temporary CSV file and return its path."""
        path = f'{tempfile.mkdtemp()}/samples.csv'
        table(smiles=SMILES, value=[0.5, 1.5, 2.5, 3.5]).to_csv(path, index=False)
        return path


class TestBuildSource(unittest.TestCase):
    def test_table_needs_a_data_column(self):
        with self.assertRaises(ValueError):
            build_source(table_file=table(smiles=SMILES, labels=[1, 0, 1, 0]))

    def test_nothing_given_is_rejected(self):
        with self.assertRaises(ValueError):
            build_source()

    def test_table_wins_over_the_label_file(self):
        source = build_source(
            label_file='ignored.csv',
            table_file=table(smiles=SMILES, labels=[1, 0, 1, 0]),
            data_column='smiles'
        )
        self.assertIsInstance(source, TableLabel)


class TestInlineLoading(unittest.TestCase):
    """The loader turns table rows into sequences in process, without touching Ray."""

    def setUp(self):
        set_alphabet('cnol')
        self.source = TableLabel(
            table(smiles=SMILES, active=[True, False, True, False]),
            'smiles',
            'active'
        )

    def tearDown(self):
        seq_manager.clear()
        reset_alphabet()

    def test_sequences_are_read_from_the_table(self):
        loader = Loader(self.source, folds=4, fold_index=0)

        self.assertTrue(loader.inline)
        self.assertEqual(len(seq_manager.train_sequences), 3)
        self.assertEqual(len(seq_manager.test_sequences), 1)

        loaded = [str(sequence) for sequence in seq_manager.train_sequences + seq_manager.test_sequences]
        self.assertEqual(sorted(loaded), sorted(smiles.lower() for smiles in SMILES))


if __name__ == '__main__':
    unittest.main()
