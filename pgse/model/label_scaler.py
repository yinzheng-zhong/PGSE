"""The standardisation a run can apply to its labels before training."""

from typing import Any, Sequence

import numpy as np


class LabelScaler:
    """The mean and standard deviation of every label, and the transform they define."""

    def __init__(self, means: Sequence[float], stds: Sequence[float]) -> None:
        """
        Args:
            means: Mean of each label.
            stds: Standard deviation of each label. Zeros are read as ones, leaving a
                constant label shifted but unscaled.
        """
        self.means: np.ndarray = np.asarray(means, dtype=np.float64).ravel()
        stds_array = np.asarray(stds, dtype=np.float64).ravel()
        self.stds: np.ndarray = np.where(stds_array == 0.0, 1.0, stds_array)

    def __len__(self) -> int:
        return len(self.means)

    def __repr__(self) -> str:
        return f'LabelScaler({len(self)} labels)'

    @classmethod
    def fit(cls, labels: np.ndarray) -> 'LabelScaler':
        """Measure the mean and standard deviation of every label.

        Args:
            labels: Training labels, one row per sample and one column per label.
        """
        matrix = np.asarray(labels, dtype=np.float64)
        matrix = matrix if matrix.ndim > 1 else matrix.reshape(-1, 1)

        return cls(matrix.mean(axis=0), matrix.std(axis=0))

    def transform(self, labels: np.ndarray) -> np.ndarray:
        """Return the labels standardised to zero mean and unit variance.

        Args:
            labels: Labels in the units of the dataset.
        """
        return ((np.asarray(labels, dtype=np.float64) - self.means) / self.stds).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Return standardised values back in the units of the dataset.

        Args:
            values: Labels or predictions on the standardised scale.
        """
        return (np.asarray(values, dtype=np.float64) * self.stds + self.means).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """The scaler as the plain lists a model's metadata file holds."""
        return {'means': self.means.tolist(), 'stds': self.stds.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> 'LabelScaler':
        """Rebuild a scaler written by to_dict.

        Args:
            values: The means and the standard deviations of the labels.
        """
        return cls(values['means'], values['stds'])
