"""The validation metrics the pipelines can score predictions with."""

import inspect
import math
from typing import Any, Optional

import numpy as np

from pgse.validation.utils import (
    ArrayLike,
    as_array,
    average_precision,
    binary_counts,
    clip_probabilities,
    confusion,
    discretise,
    is_essential_agreement,
    is_positive,
    labels_of,
    rank_average,
    roc_auc,
)


class Metric:
    """A validation metric selected by name.

    Every static method below is a metric: it takes (y_true, y_pred), returns a single score,
    and its name is the value ``--metric`` accepts. An instance holds the chosen name and the
    parameters bound to it, and is callable the way XGBoost calls a custom metric.
    """

    DEFAULT = 'essential_agreement'
    BINARY_DEFAULT = 'auroc'

    # Error metrics, where a smaller score is the better one.
    LOWER_IS_BETTER = frozenset({'rmse', 'mae', 'mape', 'log_loss'})

    def __init__(self, name: str, **params: Any) -> None:
        """
        Args:
            name: Name of one of the static methods below.
            **params: Candidate keyword arguments for the metric, e.g. ea_min and ea_max.
                Arguments the metric does not take are dropped.

        Raises:
            ValueError: If no metric of that name exists.
        """
        if name not in self.names():
            raise ValueError(
                f'Unknown validation metric {name!r}. Available metrics: {", ".join(self.names())}.'
            )

        accepted = inspect.signature(getattr(self, name)).parameters
        self.name = name
        self.params = {key: value for key, value in params.items() if key in accepted}

    @property
    def greater_is_better(self) -> bool:
        """Whether a larger score means a better model."""
        return self.name not in self.LOWER_IS_BETTER

    @classmethod
    def default_for(cls, binary: bool) -> str:
        """The metric a run scores with when none was chosen.

        Args:
            binary: Whether the run predicts a 0/1 label.
        """
        return cls.BINARY_DEFAULT if binary else cls.DEFAULT

    @classmethod
    def names(cls) -> list[str]:
        """Names of every metric, in the order they are defined."""
        return [name for name, value in vars(cls).items() if isinstance(value, staticmethod)]

    @classmethod
    def summary(cls, name: str) -> str:
        """The first line of a metric's docstring.

        Args:
            name: Name of the metric.
        """
        return (getattr(cls, name).__doc__ or '').strip().splitlines()[0]

    @classmethod
    def describe(cls) -> str:
        """A one-line summary of every metric, for command-line help."""
        return ' '.join(f'{name}: {cls.summary(name)}' for name in cls.names())

    def score(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Evaluate the metric.

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        return float(getattr(self, self.name)(y_true, y_pred, **self.params))

    def __call__(self, preds: ArrayLike, labels: Any) -> float:
        """Evaluate the metric from XGBoost's (predictions, dataset) argument order.

        Args:
            preds: Predicted values.
            labels: XGBoost DMatrix, array, or sequence holding the true labels.
        """
        return self.score(labels_of(labels), preds)

    def __repr__(self) -> str:
        return f'Metric({self.name!r}, {self.params!r})'

    @staticmethod
    def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Root mean squared error, in the units of the label.

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        true, pred = as_array(y_true), as_array(y_pred)
        return float(np.sqrt(np.mean((true - pred) ** 2)))

    @staticmethod
    def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Mean absolute error, less sensitive to outliers than RMSE.

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        true, pred = as_array(y_true), as_array(y_pred)
        return float(np.mean(np.abs(true - pred)))

    @staticmethod
    def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Mean absolute percentage error over the non-zero labels.

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        true, pred = as_array(y_true), as_array(y_pred)
        usable = true != 0.0
        if not usable.any():
            return float('nan')
        return float(np.mean(np.abs((true[usable] - pred[usable]) / true[usable])) * 100.0)

    @staticmethod
    def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Coefficient of determination, the variance of the label explained.

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        true, pred = as_array(y_true), as_array(y_pred)
        residual = float(np.sum((true - pred) ** 2))
        total = float(np.sum((true - np.mean(true)) ** 2))
        if total == 0.0:
            return 1.0 if residual == 0.0 else 0.0
        return 1.0 - residual / total

    @staticmethod
    def pearson(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Pearson correlation between label and prediction.

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        true, pred = as_array(y_true), as_array(y_pred)
        true_centred = true - np.mean(true)
        pred_centred = pred - np.mean(pred)

        denominator = math.sqrt(float(np.sum(true_centred ** 2) * np.sum(pred_centred ** 2)))
        if denominator == 0.0:
            return float('nan')
        return float(np.dot(true_centred, pred_centred) / denominator)

    @staticmethod
    def spearman(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Rank correlation, scoring monotonic agreement rather than scale.

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        return Metric.pearson(rank_average(as_array(y_true)), rank_average(as_array(y_pred)))

    @staticmethod
    def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Fraction of predictions that round to the exact label.

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        true, pred = as_array(y_true), as_array(y_pred)
        return float(np.mean(discretise(true) == discretise(pred)))

    @staticmethod
    def mcc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Matthews correlation coefficient, a balanced score in [-1, 1].

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        matrix, _ = confusion(as_array(y_true), as_array(y_pred))
        total = matrix.sum()
        if total == 0:
            return float('nan')

        true_totals = matrix.sum(axis=1)
        pred_totals = matrix.sum(axis=0)
        covariance = float(np.trace(matrix) * total - np.dot(true_totals, pred_totals))
        spread = (total ** 2 - float(np.sum(pred_totals ** 2))) * (total ** 2 - float(np.sum(true_totals ** 2)))
        if spread <= 0.0:
            return 0.0
        return covariance / math.sqrt(spread)

    @staticmethod
    def auroc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Area under the ROC curve, the chance a positive outranks a negative.

        Args:
            y_true: True 0/1 labels.
            y_pred: Predicted probabilities of the positive class.
        """
        return roc_auc(y_true, y_pred)

    @staticmethod
    def auprc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Area under the precision-recall curve, which ignores the true negatives.

        Args:
            y_true: True 0/1 labels.
            y_pred: Predicted probabilities of the positive class.
        """
        return average_precision(y_true, y_pred)

    @staticmethod
    def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Fraction of the predicted positives that are positive.

        Args:
            y_true: True 0/1 labels.
            y_pred: Predicted probabilities of the positive class, split at 0.5.
        """
        true_positive, false_positive, _, _ = binary_counts(y_true, y_pred)
        predicted = true_positive + false_positive
        return true_positive / predicted if predicted else float('nan')

    @staticmethod
    def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Fraction of the positives that are predicted positive, i.e. sensitivity.

        Args:
            y_true: True 0/1 labels.
            y_pred: Predicted probabilities of the positive class, split at 0.5.
        """
        true_positive, _, _, false_negative = binary_counts(y_true, y_pred)
        positives = true_positive + false_negative
        return true_positive / positives if positives else float('nan')

    @staticmethod
    def specificity(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Fraction of the negatives that are predicted negative.

        Args:
            y_true: True 0/1 labels.
            y_pred: Predicted probabilities of the positive class, split at 0.5.
        """
        _, false_positive, true_negative, _ = binary_counts(y_true, y_pred)
        negatives = true_negative + false_positive
        return true_negative / negatives if negatives else float('nan')

    @staticmethod
    def f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Harmonic mean of precision and recall.

        Args:
            y_true: True 0/1 labels.
            y_pred: Predicted probabilities of the positive class, split at 0.5.
        """
        true_positive, false_positive, _, false_negative = binary_counts(y_true, y_pred)
        total = 2.0 * true_positive + false_positive + false_negative
        return 2.0 * true_positive / total if total else float('nan')

    @staticmethod
    def balanced_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Mean of recall and specificity, unaffected by class imbalance.

        Args:
            y_true: True 0/1 labels.
            y_pred: Predicted probabilities of the positive class, split at 0.5.
        """
        true_positive, false_positive, true_negative, false_negative = binary_counts(y_true, y_pred)
        positives, negatives = true_positive + false_negative, true_negative + false_positive
        if not positives or not negatives:
            return float('nan')
        return (true_positive / positives + true_negative / negatives) / 2.0

    @staticmethod
    def log_loss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Mean negative log likelihood of the labels, scoring the probabilities themselves.

        Args:
            y_true: True 0/1 labels.
            y_pred: Predicted probabilities of the positive class.
        """
        true = is_positive(y_true).astype(np.float64)
        pred = clip_probabilities(y_pred)
        return float(-np.mean(true * np.log(pred) + (1.0 - true) * np.log(1.0 - pred)))

    @staticmethod
    def essential_agreement(
            y_true: ArrayLike,
            y_pred: ArrayLike,
            ea_min: Optional[float] = None,
            ea_max: Optional[float] = None
    ) -> float:
        """Fraction of predictions within one two-fold dilution of the label (log2 MIC).

        Args:
            y_true: True labels, already on a log2 scale.
            y_pred: Predicted values, already on a log2 scale.
            ea_min: Lower censoring bound, on a linear scale.
            ea_max: Upper censoring bound, on a linear scale.
        """
        return float(np.mean(is_essential_agreement(
            y_true, y_pred,
            min_after_log2=math.log2(ea_min) if ea_min else None,
            max_after_log2=math.log2(ea_max) if ea_max else None
        )))
