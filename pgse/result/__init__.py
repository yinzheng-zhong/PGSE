"""The objects a training run returns."""

from pgse.result.fold_result import FoldResult
from pgse.result.segment_importance import SegmentImportance
from pgse.result.training_result import TrainingResult

__all__ = ['FoldResult', 'SegmentImportance', 'TrainingResult']
