"""Array handling shared by the validation metrics."""

from typing import Any, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def as_array(values: ArrayLike) -> np.ndarray:
    """Return `values` as a flat float64 array.

    Args:
        values: Array or sequence of numbers.
    """
    return np.asarray(values, dtype=np.float64).ravel()


def labels_of(source: Any) -> np.ndarray:
    """Return the labels held by `source`, which may be a DMatrix or an array.

    Args:
        source: XGBoost DMatrix, array, or sequence of labels.
    """
    getter = getattr(source, 'get_label', None)
    return as_array(getter() if callable(getter) else source)


def discretise(values: np.ndarray) -> np.ndarray:
    """Round `values` to the nearest integer class.

    Args:
        values: Continuous labels or predictions.
    """
    return np.rint(values)


def rank_average(values: np.ndarray) -> np.ndarray:
    """Return the ranks of `values`, with tied entries sharing their mean rank.

    Args:
        values: Values to rank.
    """
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    means = (ends - counts + 1 + ends) / 2.0
    return means[inverse]


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the confusion matrix of the rounded inputs and the classes it is indexed by.

    Args:
        y_true: True labels.
        y_pred: Predicted values.
    """
    true_classes = discretise(y_true)
    pred_classes = discretise(y_pred)

    classes = np.union1d(np.unique(true_classes), np.unique(pred_classes))
    true_index = np.searchsorted(classes, true_classes)
    pred_index = np.searchsorted(classes, pred_classes)

    matrix = np.zeros((len(classes), len(classes)), dtype=np.float64)
    np.add.at(matrix, (true_index, pred_index), 1.0)
    return matrix, classes


def is_positive(values: ArrayLike) -> np.ndarray:
    """Return which entries of `values` are the positive class.

    Labels are 0/1 or booleans, and predictions are probabilities, so the two are split
    at the same half-way point.

    Args:
        values: True labels or predicted probabilities.
    """
    return as_array(values) >= 0.5


def clip_probabilities(values: ArrayLike, epsilon: float = 1e-15) -> np.ndarray:
    """Return `values` confined to [epsilon, 1 - epsilon].

    Args:
        values: Predicted probabilities.
        epsilon: Distance kept from 0 and 1.
    """
    return np.clip(as_array(values), epsilon, 1.0 - epsilon)


def binary_counts(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[float, float, float, float]:
    """Return the true positive, false positive, true negative and false negative counts.

    Args:
        y_true: True 0/1 labels.
        y_pred: Predicted probabilities of the positive class.
    """
    true, pred = is_positive(y_true), is_positive(y_pred)

    return (
        float(np.sum(true & pred)),
        float(np.sum(~true & pred)),
        float(np.sum(~true & ~pred)),
        float(np.sum(true & ~pred))
    )


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve, from the mean rank of the positive scores.

    Args:
        y_true: True 0/1 labels.
        y_score: Predicted scores or probabilities of the positive class.
    """
    true = is_positive(y_true)
    positives, negatives = float(np.sum(true)), float(np.sum(~true))
    if positives == 0.0 or negatives == 0.0:
        return float('nan')

    ranks = rank_average(as_array(y_score))
    return float((np.sum(ranks[true]) - positives * (positives + 1.0) / 2.0) / (positives * negatives))


def average_precision(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve, summed as a step function over recall.

    Args:
        y_true: True 0/1 labels.
        y_score: Predicted scores or probabilities of the positive class.
    """
    true = is_positive(y_true)
    positives = float(np.sum(true))
    if positives == 0.0:
        return float('nan')

    order = np.argsort(-as_array(y_score), kind='stable')
    scores = as_array(y_score)[order]

    # Keep the last sample of each run of equal scores: those are the curve's points.
    ends = np.append(np.flatnonzero(np.diff(scores)), len(scores) - 1)
    hits = np.cumsum(true[order])[ends]

    precision = hits / (ends + 1.0)
    recall = hits / positives
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def check_binary_labels(*label_sets: Optional[ArrayLike]) -> None:
    """Raise unless every label given is 0 or 1. Booleans count as 0 and 1.

    Args:
        *label_sets: Label arrays to check. Missing sets are skipped.

    Raises:
        ValueError: If any label is neither 0 nor 1.
    """
    given = [as_array(labels) for labels in label_sets if labels is not None]
    values = np.unique(np.concatenate(given or [np.empty(0)]))

    unexpected = values[(values != 0.0) & (values != 1.0)]
    if unexpected.size:
        raise ValueError(
            f'Binary mode needs 0/1 labels, but the data also holds {unexpected.tolist()}.'
        )


def is_essential_agreement(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        min_after_log2: Optional[float] = None,
        max_after_log2: Optional[float] = None
) -> np.ndarray:
    """Per-sample essential agreement of log2 labels and predictions.

    Each prediction is rounded to the nearest whole dilution on the linear scale and counts
    as agreeing when it lands within one dilution of the label. Labels at or beyond a
    censoring bound agree whenever the prediction is on the same side of that bound.

    Args:
        y_true: True labels, already on a log2 scale.
        y_pred: Predicted values, already on a log2 scale.
        min_after_log2: Lower censoring bound, on a log2 scale.
        max_after_log2: Upper censoring bound, on a log2 scale.
    """
    true, pred = as_array(y_true), as_array(y_pred)

    floor, ceil = np.floor(pred), np.ceil(pred)
    midpoint = (2.0 ** ceil + 2.0 ** floor) / 2.0
    rounded = np.where(2.0 ** pred < midpoint, floor, ceil)

    agreement = np.abs(true - rounded) <= 1
    censored_low = np.zeros(len(true), dtype=bool)

    if min_after_log2 is not None:
        censored_low = true <= min_after_log2
        agreement = np.where(censored_low, rounded <= min_after_log2, agreement)
    if max_after_log2 is not None:
        censored_high = (true >= max_after_log2) & ~censored_low
        agreement = np.where(censored_high, rounded >= max_after_log2, agreement)

    return agreement
