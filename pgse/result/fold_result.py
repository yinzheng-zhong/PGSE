"""What one fold of a training run produced."""

from typing import TYPE_CHECKING, Mapping, Optional

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
            score: float,
            label_scores: Optional[Mapping[str, float]] = None
    ) -> None:
        """
        Args:
            index: Zero-based position of the fold.
            model: The model trained on this fold, ready to predict.
            predictions: Frame with a Prediction and an Actual column per label of the
                held-out set.
            metric: Name of the validation metric the score was measured with.
            score: The metric's value on the held-out set, averaged over the labels.
            label_scores: The metric's value on each label, keyed by the label's name.
        """
        self.index: int = index
        self.model: 'PGSEModel' = model
        self.predictions: pd.DataFrame = predictions
        self.metric: str = metric
        self.score: float = score
        self.label_scores: dict[str, float] = dict(label_scores or {})

    def __repr__(self) -> str:
        if len(self.label_scores) > 1:
            per_label = ', '.join(f'{name}={value:.4f}' for name, value in self.label_scores.items())
            return f'FoldResult(index={self.index}, mean {self.metric}={self.score:.4f}, {per_label})'

        return f'FoldResult(index={self.index}, {self.metric}={self.score:.4f})'

    @property
    def segments(self) -> SegmentImportance:
        """The segments the fold's model reads, with their importance."""
        return self.model.segments
