import math
import unittest
from typing import Optional, Union

import numpy as np

from pgse.validation import Metric, is_essential_agreement


def reference_essential_agreement(
        label: Union[np.ndarray, list],
        predicted: Union[np.ndarray, list],
        min_after_log2: Optional[float] = None,
        max_after_log2: Optional[float] = None
) -> np.ndarray:
    """The per-sample essential agreement loop as it stood before the metrics were extracted.

    Args:
        label: True labels, on a log2 scale.
        predicted: Predicted values, on a log2 scale.
        min_after_log2: Lower censoring bound, on a log2 scale.
        max_after_log2: Upper censoring bound, on a log2 scale.
    """
    label = np.asarray(label)
    predicted = np.asarray(predicted)
    ea = np.zeros(len(label), dtype=bool)

    for i in range(len(label)):
        pred = predicted[i]
        ceil = np.ceil(pred)
        floor = np.floor(pred)

        mid = (2 ** ceil + 2 ** floor) / 2
        pred = floor if 2 ** pred < mid else ceil

        if min_after_log2 is not None and label[i] <= min_after_log2:
            ea[i] = pred <= min_after_log2
        elif max_after_log2 is not None and label[i] >= max_after_log2:
            ea[i] = pred >= max_after_log2
        else:
            ea[i] = abs(label[i] - pred) <= 1

    return ea


class FakeDMatrix:
    """A stand-in for an XGBoost DMatrix that only carries labels."""

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = labels

    def get_label(self) -> np.ndarray:
        return self._labels


class TestRegistry(unittest.TestCase):
    def test_default_metric_is_registered(self):
        self.assertIn(Metric.DEFAULT, Metric.names())

    def test_unknown_metric_names_the_alternatives(self):
        with self.assertRaises(ValueError) as caught:
            Metric('not_a_metric')
        self.assertIn('rmse', str(caught.exception))

    def test_every_metric_scores_a_plain_pair_of_arrays(self):
        y_true = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.9, 0.7, 0.4, 1.2, -0.2])

        for name in Metric.names():
            with self.subTest(metric=name):
                score = Metric(name).score(y_true, y_pred)
                self.assertIsInstance(score, float)
                self.assertFalse(math.isnan(score))

    def test_unrelated_parameters_are_dropped(self):
        metric = Metric('rmse', ea_min=0.5, ea_max=64)
        self.assertEqual({}, metric.params)
        self.assertEqual(0.0, metric.score([1.0, 2.0], [1.0, 2.0]))

    def test_declared_parameters_are_bound(self):
        metric = Metric('essential_agreement', ea_min=0.5, ea_max=64)
        self.assertEqual({'ea_min': 0.5, 'ea_max': 64}, metric.params)

    def test_binding_parameters_does_not_affect_later_instances(self):
        Metric('essential_agreement', ea_min=0.5)
        self.assertEqual({}, Metric('essential_agreement').params)

    def test_every_metric_is_callable_directly_off_the_class(self):
        self.assertAlmostEqual(0.0, Metric.rmse([1.0, 2.0], [1.0, 2.0]))
        self.assertAlmostEqual(1.0, Metric.essential_agreement([3.0], [3.4]))

    def test_descriptions_cover_every_metric(self):
        summary = Metric.describe()
        for name in Metric.names():
            self.assertIn(name, summary)


class TestXGBoostCallingConvention(unittest.TestCase):
    def test_labels_are_read_from_a_dmatrix(self):
        y_true = np.array([2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 3.0, 9.0])
        metric = Metric('rmse')

        self.assertEqual(
            metric.score(y_true, y_pred),
            metric(y_pred, FakeDMatrix(y_true))
        )

    def test_labels_are_read_from_a_bare_sequence(self):
        metric = Metric('mae')
        self.assertAlmostEqual(1.0, metric([2.0, 3.0], [1.0, 4.0]))


class TestRegressionMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.5, 2.5, 2.0, 4.5, 4.0])

    def test_rmse(self):
        expected = math.sqrt(np.mean((self.y_true - self.y_pred) ** 2))
        self.assertAlmostEqual(expected, Metric('rmse').score(self.y_true, self.y_pred))

    def test_mae(self):
        self.assertAlmostEqual(0.7, Metric('mae').score(self.y_true, self.y_pred))

    def test_mape_ignores_zero_labels(self):
        score = Metric('mape').score([0.0, 2.0], [5.0, 3.0])
        self.assertAlmostEqual(50.0, score)

    def test_mape_is_nan_when_every_label_is_zero(self):
        self.assertTrue(math.isnan(Metric('mape').score([0.0, 0.0], [1.0, 2.0])))

    def test_r2_of_a_perfect_fit(self):
        self.assertAlmostEqual(1.0, Metric('r2').score(self.y_true, self.y_true))

    def test_r2_of_the_mean_predictor(self):
        mean = np.full_like(self.y_true, self.y_true.mean())
        self.assertAlmostEqual(0.0, Metric('r2').score(self.y_true, mean))

    def test_r2_matches_the_definition(self):
        residual = np.sum((self.y_true - self.y_pred) ** 2)
        total = np.sum((self.y_true - self.y_true.mean()) ** 2)
        self.assertAlmostEqual(1 - residual / total, Metric('r2').score(self.y_true, self.y_pred))

    def test_pearson_matches_numpy(self):
        expected = float(np.corrcoef(self.y_true, self.y_pred)[0, 1])
        self.assertAlmostEqual(expected, Metric('pearson').score(self.y_true, self.y_pred))

    def test_pearson_is_nan_for_a_constant_prediction(self):
        score = Metric('pearson').score(self.y_true, np.ones_like(self.y_true))
        self.assertTrue(math.isnan(score))

    def test_spearman_is_one_for_any_increasing_prediction(self):
        monotonic = np.exp(self.y_true)
        self.assertAlmostEqual(1.0, Metric('spearman').score(self.y_true, monotonic))

    def test_spearman_averages_tied_ranks(self):
        score = Metric('spearman').score([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 3.0])
        self.assertAlmostEqual(0.9486832980505138, score)


class TestClassificationMetrics(unittest.TestCase):
    def test_accuracy_rounds_both_sides(self):
        self.assertAlmostEqual(0.75, Metric('accuracy').score([0, 1, 1, 0], [0.4, 0.6, 0.2, 0.1]))

    def test_mcc_of_a_perfect_prediction(self):
        self.assertAlmostEqual(1.0, Metric('mcc').score([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]))

    def test_mcc_of_an_inverted_prediction(self):
        self.assertAlmostEqual(-1.0, Metric('mcc').score([0, 1, 0, 1], [1.0, 0.0, 1.0, 0.0]))

    def test_mcc_is_zero_for_a_constant_prediction(self):
        self.assertAlmostEqual(0.0, Metric('mcc').score([0, 1, 0, 1], [0.0, 0.0, 0.0, 0.0]))


class TestEssentialAgreement(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.labels = rng.integers(-2, 8, size=200).astype(float)
        self.predictions = self.labels + rng.normal(0.0, 1.5, size=200)

    def test_matches_the_previous_implementation(self):
        for bounds in [(None, None), (0.0, None), (None, 6.0), (0.0, 6.0), (2.0, 2.0)]:
            with self.subTest(bounds=bounds):
                np.testing.assert_array_equal(
                    reference_essential_agreement(self.labels, self.predictions, *bounds),
                    is_essential_agreement(self.labels, self.predictions, *bounds)
                )

    def test_matches_the_previous_rate_through_the_metric(self):
        expected = float(np.mean(reference_essential_agreement(
            self.labels, self.predictions,
            min_after_log2=math.log2(0.5), max_after_log2=math.log2(64)
        )))
        metric = Metric('essential_agreement', ea_min=0.5, ea_max=64)
        self.assertAlmostEqual(expected, metric.score(self.labels, self.predictions))

    def test_bounds_are_converted_from_the_linear_scale(self):
        censored = Metric('essential_agreement', ea_max=64).score([6.0, 6.0], [9.0, 3.0])
        uncensored = Metric('essential_agreement').score([6.0, 6.0], [9.0, 3.0])
        self.assertAlmostEqual(0.5, censored)
        self.assertAlmostEqual(0.0, uncensored)

    def test_a_zero_bound_is_treated_as_absent(self):
        metric = Metric('essential_agreement', ea_min=0, ea_max=0)
        self.assertAlmostEqual(1.0, metric.score([3.0], [3.4]))

    def test_predictions_round_on_the_linear_scale(self):
        metric = Metric('essential_agreement')
        self.assertAlmostEqual(1.0, metric.score([1.0], [2.5]))
        self.assertAlmostEqual(0.0, metric.score([1.0], [2.9]))


if __name__ == '__main__':
    unittest.main()
