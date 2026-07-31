"""
PGSE's public entry points.

The pipelines are imported on first attribute access (PEP 562), so ``import pgse``
does not import the training stack (Ray, XGBoost, scikit-learn).
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Read by type checkers only; not executed at runtime.
    from pgse.pipeline.pgse_pipeline import Pipeline as TrainingPipeline
    from pgse.pipeline.pgse_inference_pipeline import Pipeline as InferencePipeline
    from pgse.pipeline.regular_pipline import Pipeline as PureXGBPipeline
    from pgse.pipeline.pgse_inference_pipeline_web import Pipeline as InferencePipelineWeb

# Exported name -> the module whose ``Pipeline`` class it refers to.
_PIPELINES = {
    'TrainingPipeline': 'pgse.pipeline.pgse_pipeline',
    'InferencePipeline': 'pgse.pipeline.pgse_inference_pipeline',
    'PureXGBPipeline': 'pgse.pipeline.regular_pipline',
    'InferencePipelineWeb': 'pgse.pipeline.pgse_inference_pipeline_web',
}

__all__ = ["TrainingPipeline", "InferencePipeline", "PureXGBPipeline", "InferencePipelineWeb"]


def __getattr__(name: str) -> Any:
    """Import a pipeline on first access and cache it in the module globals."""
    module_name = _PIPELINES.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    pipeline = import_module(module_name).Pipeline
    globals()[name] = pipeline  # Later lookups skip __getattr__ entirely.
    return pipeline


def __dir__() -> list:
    return sorted(set(globals()) | set(_PIPELINES))
