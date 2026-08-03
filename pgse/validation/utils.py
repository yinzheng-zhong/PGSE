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
