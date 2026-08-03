"""The segments a run discovered, paired with their importance."""

from typing import Iterator, Sequence, Union

import numpy as np
import pandas as pd

SEGMENT_COLUMN = 'Segment'
IMPORTANCE_COLUMN = 'Importance'

Importances = Union[Sequence[float], np.ndarray]


class SegmentImportance:
    """Segments in count-matrix column order, each with the importance a model gave it."""

    def __init__(self, segments: Sequence[str], importances: Importances) -> None:
        """
        Args:
            segments: The segments, ordered as the columns of the count matrix.
            importances: One importance per segment, in the same order.
        """
        if len(segments) != len(importances):
            raise ValueError(
                f'Got {len(segments)} segments but {len(importances)} importances.'
            )

        self.segments: list[str] = list(segments)
        self.importances: np.ndarray = np.asarray(importances, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterator[tuple[str, float]]:
        return zip(self.segments, (float(value) for value in self.importances))

    def __getitem__(self, index: int) -> tuple[str, float]:
        return self.segments[index], float(self.importances[index])

    def __repr__(self) -> str:
        return f'SegmentImportance({len(self)} segments, {int(np.count_nonzero(self.importances))} used)'

    def to_frame(self) -> pd.DataFrame:
        """The segments and their importance as a two-column frame, in column order."""
        return pd.DataFrame({SEGMENT_COLUMN: self.segments, IMPORTANCE_COLUMN: self.importances})

    def ranked(self) -> 'SegmentImportance':
        """The same segments, ordered from most to least important."""
        order = np.argsort(-self.importances, kind='stable')
        return SegmentImportance([self.segments[i] for i in order], self.importances[order])

    def top(self, n: int = 20) -> 'SegmentImportance':
        """The n most important segments.

        Args:
            n: How many segments to keep.
        """
        ranked = self.ranked()
        return SegmentImportance(ranked.segments[:n], ranked.importances[:n])

    def to_csv(self, filename: str) -> None:
        """Write the segments and their importance to a CSV file.

        Args:
            filename: Path of the file to write.
        """
        self.to_frame().to_csv(filename, index=False)

    @classmethod
    def from_csv(cls, filename: str) -> 'SegmentImportance':
        """Read segments written by to_csv, treating a missing importance column as zeros.

        Args:
            filename: Path of the file to read.
        """
        frame = pd.read_csv(filename)
        segments = frame[SEGMENT_COLUMN].astype(str).tolist()
        if IMPORTANCE_COLUMN in frame:
            return cls(segments, frame[IMPORTANCE_COLUMN].to_numpy(dtype=np.float64))
        return cls(segments, np.zeros(len(segments)))

    @classmethod
    def from_xgb_importance(
            cls,
            segments: Sequence[str],
            importance: pd.DataFrame
    ) -> 'SegmentImportance':
        """Pair each segment with the score XGBoost reported for its column, zero if unused.

        Args:
            segments: The segments, ordered as the columns of the count matrix.
            importance: Frame with a Feature column of column indices and an
                Importance column of scores.
        """
        scores = np.zeros(len(segments), dtype=np.float64)
        for feature, score in zip(importance['Feature'], importance['Importance']):
            column = int(feature)
            if 0 <= column < len(segments):
                scores[column] = float(score)

        return cls(segments, scores)

    @staticmethod
    def merge(parts: Sequence['SegmentImportance']) -> 'SegmentImportance':
        """Combine several sets into one, averaging each segment's importance over them.

        Args:
            parts: The sets to combine, e.g. one per fold.
        """
        totals: dict[str, float] = {}
        for part in parts:
            for segment, importance in part:
                totals[segment] = totals.get(segment, 0.0) + importance

        divisor = max(len(parts), 1)
        merged = SegmentImportance(list(totals), [total / divisor for total in totals.values()])
        return merged.ranked()
