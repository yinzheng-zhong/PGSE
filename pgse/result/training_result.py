"""What a whole training run produced."""

from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
import pandas as pd

from pgse.result.fold_result import FoldResult
from pgse.result.segment_importance import SegmentImportance

if TYPE_CHECKING:  # Read by type checkers only, so importing a result does not load XGBoost.
    from pgse.model.pgse_model import PGSEModel


class TrainingResult:
    """Every fold of a training run: the models, the segments they found and their scores."""

    def __init__(
            self,
            folds: Sequence[FoldResult],
            metric: str,
            greater_is_better: bool = True
    ) -> None:
        """
        Args:
            folds: One result per fold, in the order they were trained.
            metric: Name of the validation metric the folds were scored with.
            greater_is_better: Whether a larger score means a better model.
        """
        self.folds: list[FoldResult] = list(folds)
        self.metric: str = metric
        self.greater_is_better: bool = greater_is_better

    def __len__(self) -> int:
        return len(self.folds)

    def __iter__(self) -> Iterator[FoldResult]:
        return iter(self.folds)

    def __getitem__(self, index: int) -> FoldResult:
        return self.folds[index]

    def __repr__(self) -> str:
        return f'TrainingResult({len(self)} folds, mean {self.metric}={self.score:.4f})'

    @property
    def models(self) -> list['PGSEModel']:
        """One trained model per fold."""
        return [fold.model for fold in self.folds]

    @property
    def model(self) -> 'PGSEModel':
        """The model of the best-scoring fold."""
        return self.best_fold.model

    @property
    def best_fold(self) -> FoldResult:
        """The fold with the best score under the run's metric."""
        if not self.folds:
            raise ValueError('The run produced no folds.')

        return max(self.folds, key=lambda fold: fold.score if self.greater_is_better else -fold.score)

    @property
    def scores(self) -> list[float]:
        """The score of each fold."""
        return [fold.score for fold in self.folds]

    @property
    def score(self) -> float:
        """The mean score across the folds."""
        return float(np.mean(self.scores)) if self.folds else float('nan')

    @property
    def label_scores(self) -> dict[str, float]:
        """The mean score of every label across the folds, empty for a single label."""
        names = list(self.folds[0].label_scores) if self.folds else []
        if len(names) < 2:
            return {}

        return {
            name: float(np.mean([fold.label_scores[name] for fold in self.folds]))
            for name in names
        }

    @property
    def segments(self) -> SegmentImportance:
        """Every discovered segment, ranked by its mean importance across the folds."""
        return SegmentImportance.merge([fold.segments for fold in self.folds])

    @property
    def predictions(self) -> pd.DataFrame:
        """The held-out predictions of every fold, with the fold each row came from."""
        frames = [fold.predictions.assign(Fold=fold.index) for fold in self.folds]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def to_frame(self) -> pd.DataFrame:
        """One row per fold: its index, score, per-label scores and how many segments it kept."""
        columns: dict[str, list] = {
            'Fold': [fold.index for fold in self.folds],
            self.metric: self.scores,
        }

        for name in self.label_scores:
            columns[f'{self.metric}_{name}'] = [fold.label_scores[name] for fold in self.folds]

        columns['Segments'] = [len(fold.segments) for fold in self.folds]
        return pd.DataFrame(columns)
