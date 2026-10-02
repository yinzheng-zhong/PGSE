"""Helpers turning the labels of a dataset into the array the pipelines train on."""

from typing import Sequence, Union

import numpy as np
import pandas as pd

MAX_REPORTED_VALUES = 5

# The name of one label column, or the names of several of them.
LabelColumns = Union[str, Sequence[str]]


def as_label_columns(columns: LabelColumns) -> list[str]:
    """Return the label columns as a list of names.

    Args:
        columns: Name of a single label column, or a sequence of names.

    Raises:
        ValueError: If no column is given, or a name is repeated.
    """
    names = [columns] if isinstance(columns, str) else [str(name) for name in columns]

    if not names:
        raise ValueError('At least one label column is needed.')

    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f'The label columns {repeated} are given more than once.')

    return names


def to_float_labels(labels: pd.Series, column: str) -> np.ndarray:
    """Convert a label column into a float32 array, one entry per sample.

    Args:
        labels: The column holding the target value of each sample. Numbers, numeric
            strings and booleans are all accepted.
        column: Name of the column, used in the error message.
    """
    numeric = labels if labels.dtype == bool else pd.to_numeric(labels, errors='coerce')

    invalid = pd.isna(numeric)
    if invalid.any():
        offending = sorted({str(value) for value in labels[invalid]})[:MAX_REPORTED_VALUES]
        raise ValueError(f'Column {column!r} holds {int(invalid.sum())} values that are not numbers: {offending}.')

    return np.asarray(numeric, dtype=np.float32)


def to_float_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Convert the label columns into a float32 matrix, one row per sample and one column per label.

    Args:
        frame: The table holding the samples.
        columns: Names of the label columns, in the order they are read.
    """
    if not len(frame):
        return np.zeros((0, len(columns)), dtype=np.float32)

    return np.stack([to_float_labels(frame[column], column) for column in columns], axis=1)


def missing_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Return the columns that the frame does not hold.

    Args:
        frame: The table to check.
        columns: Names the table is expected to hold.
    """
    return [column for column in columns if column not in frame.columns]
