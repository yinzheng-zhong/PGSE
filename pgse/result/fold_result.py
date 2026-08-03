"""What one fold of a training run produced."""

from typing import TYPE_CHECKING

import pandas as pd

from pgse.result.segment_importance import SegmentImportance

if TYPE_CHECKING:  # Read by type checkers only, so importing a fold does not load XGBoost.
    from pgse.model.pgse_model import PGSEModel


class FoldResult:
    """The model, predictions and score of a single fold."""

    def __init__(
            self,
            index: int,
            model: 'PGSEModel',
            predictions: pd.DataFrame,
            metric: str,
            score: float
    ) -> None:
        """
        Args:
            index: Zero-based position of the fold.
            model: The model trained on this fold, ready to predict.
            predictions: Frame with a Prediction and an Actual column for the held-out set.
            metric: Name of the validation metric the score was measured with.
            score: The metric's value on the held-out set.
        """
        self.index: int = index
        self.model: 'PGSEModel' = model
        self.predictions: pd.DataFrame = predictions
        self.metric: str = metric
        self.score: float = score

    def __repr__(self) -> str:
        return f'FoldResult(index={self.index}, {self.metric}={self.score:.4f})'

    @property
    def segments(self) -> SegmentImportance:
        """The segments the fold's model reads, with their importance."""
        return self.model.segments
