import unittest

import numpy as np
import pandas as pd
import xgboost as xgb

from pgse.dataset.label_utils import as_label_columns, to_float_matrix
from pgse.model.label_scaler import LabelScaler
from pgse.model.xgb import XGBoost
from pgse.result.predictions import (
    as_matrix,
    as_target,
    actual_columns,
    prediction_columns,
    result_frame,
    to_matrices,
)
from pgse.validation import Metric

# Features 0-4 drive the first label, 10-14 the second one.
FIRST_LABEL_FEATURES = list(range(5))
SECOND_LABEL_FEATURES = list(range(10, 15))
N_FEATURES = 30


def planted_dataset(second_label_scale: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Build a dataset whose two labels are driven by different features.

    Args:
        second_label_scale: Scale of the second label, relative to the first one.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, N_FEATURES))
    y = np.stack([
        10.0 * x[:, FIRST_LABEL_FEATURES].sum(axis=1),
        second_label_scale * x[:, SECOND_LABEL_FEATURES].sum(axis=1),
    ], axis=1)

    return x, y


def train_booster(x: np.ndarray, y: np.ndarray) -> xgb.Booster:
    """Train a multi-output booster with the parameters the pipeline uses.

    Args:
        x: The feature matrix.
        y: The labels, one column per label.
    """
    model = XGBoost(partition_size_target=0, boost_rounds=60, num_cpu_per_node=1)
    return xgb.train(model.params, xgb.DMatrix(x, label=y), 60)


class TestLabelColumns(unittest.TestCase):
    def test_a_single_name_becomes_a_list(self):
        self.assertEqual(['mic'], as_label_columns('mic'))

    def test_several_names_are_kept_in_order(self):
        self.assertEqual(['mic', 'growth'], as_label_columns(['mic', 'growth']))

    def test_no_name_is_rejected(self):
        with self.assertRaises(ValueError):
            as_label_columns([])

    def test_a_repeated_name_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            as_label_columns(['mic', 'mic'])
        self.assertIn('mic', str(caught.exception))

    def test_a_matrix_holds_one_column_per_label(self):
        frame = pd.DataFrame({'mic': ['2', '8'], 'growth': [0.5, 1.5]})
        np.testing.assert_allclose([[2.0, 0.5], [8.0, 1.5]], to_float_matrix(frame, ['mic', 'growth']))


class TestPredictionFrame(unittest.TestCase):
    def test_a_single_label_keeps_the_plain_columns(self):
        self.assertEqual(['Prediction'], prediction_columns(['mic']))
        self.assertEqual(['Actual'], actual_columns(['mic']))

    def test_several_labels_are_named_per_label(self):
        self.assertEqual(['Prediction_mic', 'Prediction_growth'], prediction_columns(['mic', 'growth']))
        self.assertEqual(['Actual_mic', 'Actual_growth'], actual_columns(['mic', 'growth']))

    def test_a_frame_round_trips_through_its_matrices(self):
        actual = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = np.array([[1.5, 2.5], [3.5, 4.5]])

        frame = result_frame(actual, predicted, ['mic', 'growth'])
        read_actual, read_predicted = to_matrices(frame, ['mic', 'growth'])

        np.testing.assert_allclose(actual, read_actual)
        np.testing.assert_allclose(predicted, read_predicted)

    def test_a_single_label_frame_round_trips(self):
        frame = result_frame(np.array([[1.0], [3.0]]), np.array([1.5, 3.5]), ['mic'])

        self.assertEqual(['Prediction', 'Actual'], list(frame.columns))
        np.testing.assert_allclose([[1.0], [3.0]], to_matrices(frame, ['mic'])[0])

    def test_one_label_is_flattened_for_xgboost(self):
        self.assertEqual((2,), as_target(np.array([[1.0], [2.0]])).shape)
        self.assertEqual((2, 2), as_target(np.array([[1.0, 2.0], [3.0, 4.0]])).shape)
        self.assertEqual((2, 1), as_matrix(np.array([1.0, 2.0])).shape)


class TestLabelScaler(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([[2.0, 0.5], [8.0, 1.5], [4.0, 1.0]])
        self.scaler = LabelScaler.fit(self.labels)

    def test_the_labels_are_standardised(self):
        scaled = self.scaler.transform(self.labels)

        np.testing.assert_allclose([0.0, 0.0], scaled.mean(axis=0), atol=1e-6)
        np.testing.assert_allclose([1.0, 1.0], scaled.std(axis=0), atol=1e-6)

    def test_the_transform_is_undone(self):
        np.testing.assert_allclose(
            self.labels,
            self.scaler.inverse_transform(self.scaler.transform(self.labels)),
            rtol=1e-5
        )

    def test_a_constant_label_survives(self):
        scaler = LabelScaler.fit(np.array([[3.0], [3.0]]))

        np.testing.assert_allclose([[0.0], [0.0]], scaler.transform(np.array([[3.0], [3.0]])))
        np.testing.assert_allclose([[3.0]], scaler.inverse_transform(np.array([[0.0]])))

    def test_it_round_trips_through_a_dict(self):
        restored = LabelScaler.from_dict(self.scaler.to_dict())

        np.testing.assert_allclose(self.scaler.means, restored.means)
        np.testing.assert_allclose(self.scaler.stds, restored.stds)


class TestMetricPerLabel(unittest.TestCase):
    def setUp(self):
        self.true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        self.predicted = np.array([[1.0, 12.0], [2.0, 22.0], [3.0, 32.0]])

    def test_every_label_is_scored(self):
        scores = Metric('rmse').score_each(self.true, self.predicted)

        self.assertEqual(2, len(scores))
        self.assertAlmostEqual(0.0, scores[0])
        self.assertAlmostEqual(2.0, scores[1])

    def test_the_headline_score_is_their_mean(self):
        self.assertAlmostEqual(1.0, Metric('rmse').score(self.true, self.predicted))

    def test_a_single_label_is_scored_as_before(self):
        self.assertAlmostEqual(2.0, Metric('rmse').score(np.array([10.0, 20.0]), np.array([12.0, 22.0])))
        self.assertEqual(1, len(Metric('rmse').score_each(np.array([10.0]), np.array([12.0]))))


class TestPerLabelImportance(unittest.TestCase):
    """Feature selection has to see the features of every label, whatever the label's scale."""

    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = planted_dataset()
        cls.booster = train_booster(cls.x, cls.y)
        cls.model = XGBoost(partition_size_target=0, boost_rounds=60, num_cpu_per_node=1)
        cls.indices = np.arange(N_FEATURES)
        cls.entries = cls.model._partition_importance(cls.booster, cls.indices, 2)

    def ranking(self, entries, n_labels: int) -> list[int]:
        """The features of an importance frame, most important first.

        Args:
            entries: The (feature, label, importance) triples to rank.
            n_labels: Number of labels the model predicts.
        """
        frame = self.model._importance_frame(entries, n_labels)
        return [int(feature) for feature in frame['Feature']]

    def test_every_label_reports_its_own_features(self):
        by_label = {label: set() for label in (0, 1)}
        for feature, label, _ in self.entries:
            by_label[label].add(feature)

        self.assertTrue(set(FIRST_LABEL_FEATURES) <= by_label[0])
        self.assertTrue(set(SECOND_LABEL_FEATURES) <= by_label[1])

    def test_the_small_scale_label_keeps_its_features(self):
        top = self.ranking(self.entries, 2)[:10]

        self.assertEqual(set(FIRST_LABEL_FEATURES) | set(SECOND_LABEL_FEATURES), set(top))

    def test_pooling_the_raw_gains_would_bury_them(self):
        pooled = [(feature, 0, gain) for feature, _, gain in self.entries]
        top = self.ranking(pooled, 1)[:10]

        self.assertTrue(set(FIRST_LABEL_FEATURES) <= set(top))
        self.assertFalse(set(SECOND_LABEL_FEATURES) <= set(top))

    def test_each_label_carries_the_same_total_importance(self):
        frame = self.model._importance_frame(self.entries, 2)

        self.assertAlmostEqual(2.0, float(frame['Importance'].sum()), places=5)

    def test_a_single_label_keeps_its_raw_gains(self):
        entries = [(3, 0, 2.0), (7, 0, 8.0)]
        frame = self.model._importance_frame(entries, 1)

        self.assertEqual([7, 3], [int(feature) for feature in frame['Feature']])
        self.assertEqual([8.0, 2.0], list(frame['Importance']))


if __name__ == '__main__':
    unittest.main()
