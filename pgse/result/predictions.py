"""The shape and the column names labels and predictions are reported with."""

from typing import Sequence

import numpy as np
import pandas as pd

PREDICTION_COLUMN = 'Prediction'
ACTUAL_COLUMN = 'Actual'


def as_matrix(values: np.ndarray) -> np.ndarray:
    """Return the values with one row per sample and one column per label.

    Args:
        values: Labels or predictions, either a matrix already or one label's column.
    """
    array = np.asarray(values)
    return array if array.ndim > 1 else array.reshape(-1, 1)


def as_target(labels: np.ndarray) -> np.ndarray:
    """Return the labels in the shape XGBoost takes them: a column per label, flat for one.

    Args:
        labels: Labels with one row per sample and one column per label.
    """
    matrix = as_matrix(labels)
    return matrix[:, 0] if matrix.shape[1] == 1 else matrix


def prediction_columns(label_names: Sequence[str]) -> list[str]:
    """The name of the prediction column of every label.

    Args:
        label_names: Name of each label, in the order of the label columns.
    """
    return _columns(PREDICTION_COLUMN, label_names)


def actual_columns(label_names: Sequence[str]) -> list[str]:
    """The name of the true-label column of every label.

    Args:
        label_names: Name of each label, in the order of the label columns.
    """
    return _columns(ACTUAL_COLUMN, label_names)


def result_frame(
        actual: np.ndarray,
        predicted: np.ndarray,
        label_names: Sequence[str]
) -> pd.DataFrame:
    """Pair every prediction with its label, one column pair per label.

    Args:
        actual: True labels, one row per sample and one column per label.
        predicted: Predictions, in the same shape as actual.
        label_names: Name of each label, in the order of the label columns.
    """
    actual_matrix, predicted_matrix = as_matrix(actual), as_matrix(predicted)

    columns = {}
    for index, (prediction_name, actual_name) in enumerate(
            zip(prediction_columns(label_names), actual_columns(label_names))
    ):
        columns[prediction_name] = predicted_matrix[:, index]
        columns[actual_name] = actual_matrix[:, index]

    return pd.DataFrame(columns)


def to_matrices(frame: pd.DataFrame, label_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read the true labels and the predictions of a result frame back out of it.

    Args:
        frame: A frame built by result_frame.
        label_names: Name of each label, in the order of the label columns.
    """
    return (
        frame[actual_columns(label_names)].to_numpy(dtype=np.float64),
        frame[prediction_columns(label_names)].to_numpy(dtype=np.float64)
    )


def _columns(prefix: str, label_names: Sequence[str]) -> list[str]:
    """The column of every label, left unsuffixed when there is only one label.

    Args:
        prefix: Prediction or Actual.
        label_names: Name of each label, in the order of the label columns.
    """
    if len(label_names) < 2:
        return [prefix]

    return [f'{prefix}_{name}' for name in label_names]
