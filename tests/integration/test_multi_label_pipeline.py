import os
import unittest
import unittest.mock
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from pgse import PGSEModel, TrainingPipeline
from pgse.dataset.alphabet import reset_alphabet

# Each motif drives one of the labels, and the labels are on very different scales.
FIRST_MOTIF = 'aaggttcc'
SECOND_MOTIF = 'ttccggaa'
LABEL_NAMES = ['mic', 'growth']


def samples(count: int = 60) -> pd.DataFrame:
    """Build a table of sequences whose two labels are driven by different motifs.

    Args:
        count: Number of samples to build.
    """
    rng = np.random.default_rng(7)
    rows = []

    for index in range(count):
        background = ''.join(rng.choice(list('atgc'), size=120))
        first = bool(index % 2)
        second = bool((index // 2) % 2)

        sequence = background
        if first:
            sequence = sequence[:20] + FIRST_MOTIF + sequence[20:]
        if second:
            sequence = sequence[:80] + SECOND_MOTIF + sequence[80:]

        rows.append({
            'sequence': sequence,
            'mic': 100.0 if first else 20.0,
            'growth': 0.05 if second else 0.01,
        })

    return pd.DataFrame(rows)


def train(table: pd.DataFrame, label_columns=LABEL_NAMES, **kwargs):
    """Run a short multi-label pipeline over the table.

    Args:
        table: The samples to train on.
        label_columns: The label columns to train on.
        kwargs: Extra pipeline arguments.
    """
    pipeline = TrainingPipeline(
        table_file=table,
        data_column='sequence',
        label_columns=label_columns,
        k=4,
        ext=2,
        target=8,
        num_rounds=40,
        metric='rmse',
        workers=2,
        **kwargs
    )
    return pipeline.train()


class TestMultiLabelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = samples()
        cls.result = train(cls.table)
        cls.model = cls.result.model

    @classmethod
    def tearDownClass(cls):
        reset_alphabet()

    def test_every_label_is_predicted(self):
        predictions = self.model.predict_sequences(self.table['sequence'].tolist())

        self.assertEqual((len(self.table), 2), predictions.shape)
        self.assertTrue(np.isfinite(predictions).all())

    def test_the_model_records_its_labels(self):
        self.assertEqual(LABEL_NAMES, self.model.label_names)

    def test_the_fold_frame_holds_a_column_pair_per_label(self):
        predictions = self.result.folds[0].predictions

        for name in LABEL_NAMES:
            self.assertIn(f'Prediction_{name}', predictions)
            self.assertIn(f'Actual_{name}', predictions)

    def test_every_label_is_scored_and_averaged(self):
        fold = self.result.folds[0]

        self.assertEqual(set(LABEL_NAMES), set(fold.label_scores))
        self.assertAlmostEqual(float(np.mean(list(fold.label_scores.values()))), fold.score, places=5)

    def test_the_run_reports_the_score_of_every_label(self):
        frame = self.result.to_frame()

        self.assertIn('rmse_mic', frame)
        self.assertIn('rmse_growth', frame)
        self.assertEqual(set(LABEL_NAMES), set(self.result.label_scores))

    def test_the_wider_label_is_predicted_from_its_own_motif(self):
        predictions = self.model.predict_sequences(self.table['sequence'].tolist())
        high = self.table['mic'] > self.table['mic'].min()

        self.assertGreater(predictions[high, 0].mean(), predictions[~high, 0].mean())

    def test_labels_on_far_apart_scales_are_warned_about(self):
        with unittest.mock.patch('pgse.pipeline.pgse_pipeline.logger') as log:
            TrainingPipeline(
                table_file=self.table, data_column='sequence', label_columns=LABEL_NAMES
            )

        warning = ' '.join(str(call) for call in log.warning.call_args_list)
        self.assertIn('standardise_labels', warning)
        self.assertIn('growth', warning)

    def test_the_labels_survive_a_save_and_load(self):
        with TemporaryDirectory() as tmp_dir:
            self.model.save(os.path.join(tmp_dir, 'model'))
            loaded = PGSEModel.load(os.path.join(tmp_dir, 'model'))

            self.assertEqual(LABEL_NAMES, loaded.label_names)
            np.testing.assert_allclose(
                self.model.predict_sequences(self.table['sequence'].tolist()),
                loaded.predict_sequences(self.table['sequence'].tolist())
            )


class TestStandardisedLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = samples()
        cls.result = train(cls.table, standardise_labels=True)
        cls.model = cls.result.model

    @classmethod
    def tearDownClass(cls):
        reset_alphabet()

    def test_the_predictions_stay_in_the_units_of_the_labels(self):
        predictions = self.model.predict_sequences(self.table['sequence'].tolist())

        for index, name in enumerate(LABEL_NAMES):
            with self.subTest(label=name):
                column = predictions[:, index]
                self.assertGreater(column.mean(), self.table[name].min() / 2.0)
                self.assertLess(column.mean(), self.table[name].max() * 2.0)

    def test_the_scaler_survives_a_save_and_load(self):
        with TemporaryDirectory() as tmp_dir:
            self.model.save(os.path.join(tmp_dir, 'model'))
            loaded = PGSEModel.load(os.path.join(tmp_dir, 'model'))

            self.assertIsNotNone(loaded.scaler)
            np.testing.assert_allclose(
                self.model.predict_sequences(self.table['sequence'].tolist()),
                loaded.predict_sequences(self.table['sequence'].tolist())
            )

    def test_every_label_is_learned_from_its_own_motif(self):
        predictions = self.model.predict_sequences(self.table['sequence'].tolist())

        for index, name in enumerate(LABEL_NAMES):
            with self.subTest(label=name):
                high = self.table[name] > self.table[name].min()
                self.assertGreater(predictions[high, index].mean(), predictions[~high, index].mean())

    def test_no_scale_warning_is_raised(self):
        with unittest.mock.patch('pgse.pipeline.pgse_pipeline.logger') as log:
            TrainingPipeline(
                table_file=self.table, data_column='sequence', label_columns=LABEL_NAMES,
                standardise_labels=True
            )

        self.assertEqual([], log.warning.call_args_list)


class TestSingleLabelIsUnchanged(unittest.TestCase):
    """One label keeps the plain column names and the flat predictions."""

    @classmethod
    def tearDownClass(cls):
        reset_alphabet()

    def test_a_single_label_run_reports_the_old_columns(self):
        result = train(samples(), label_columns='mic')
        predictions = result.folds[0].predictions

        self.assertEqual(['Prediction', 'Actual'], list(predictions.columns))
        self.assertEqual({}, result.label_scores)
        self.assertEqual((len(samples()),), result.model.predict_sequences(samples()['sequence'].tolist()).shape)


if __name__ == '__main__':
    unittest.main()
