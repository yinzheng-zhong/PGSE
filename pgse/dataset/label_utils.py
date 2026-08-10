"""Helpers turning the labels of a dataset into the array the pipelines train on."""

import numpy as np
import pandas as pd

MAX_REPORTED_VALUES = 5


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
