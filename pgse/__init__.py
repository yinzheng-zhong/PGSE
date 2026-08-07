"""
PGSE's public entry points.

The pipelines and the model are imported on first attribute access (PEP 562), so
``import pgse`` does not import the training stack (Ray, XGBoost, scikit-learn).
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Read by type checkers only; not executed at runtime.
    from pgse.pipeline.pgse_pipeline import Pipeline as TrainingPipeline
    from pgse.pipeline.pgse_inference_pipeline import Pipeline as InferencePipeline
    from pgse.pipeline.regular_pipline import Pipeline as PureXGBPipeline
    from pgse.pipeline.pgse_inference_pipeline_web import Pipeline as InferencePipelineWeb
    from pgse.dataset.alphabet import Alphabet
    from pgse.model.pgse_model import PGSEModel
    from pgse.result.fold_result import FoldResult
    from pgse.result.segment_importance import SegmentImportance
    from pgse.result.training_result import TrainingResult
    from pgse.validation.metrics import Metric

# Exported name -> the module and the attribute it refers to.
_EXPORTS = {
    'TrainingPipeline': ('pgse.pipeline.pgse_pipeline', 'Pipeline'),
    'InferencePipeline': ('pgse.pipeline.pgse_inference_pipeline', 'Pipeline'),
    'PureXGBPipeline': ('pgse.pipeline.regular_pipline', 'Pipeline'),
    'InferencePipelineWeb': ('pgse.pipeline.pgse_inference_pipeline_web', 'Pipeline'),
    'PGSEModel': ('pgse.model.pgse_model', 'PGSEModel'),
    'TrainingResult': ('pgse.result.training_result', 'TrainingResult'),
    'FoldResult': ('pgse.result.fold_result', 'FoldResult'),
    'SegmentImportance': ('pgse.result.segment_importance', 'SegmentImportance'),
    'Alphabet': ('pgse.dataset.alphabet', 'Alphabet'),
    'Metric': ('pgse.validation.metrics', 'Metric'),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import an entry point on first access and cache it in the module globals."""
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    module_name, attribute = export
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value  # Later lookups skip __getattr__ entirely.
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_EXPORTS))
