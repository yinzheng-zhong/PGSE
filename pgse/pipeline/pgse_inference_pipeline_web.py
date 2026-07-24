from pgse.dataset.loader_inference import LoaderInference
import numpy as np
import xgboost as xgb
from pgse.etc.alphabet import AUTO, AlphabetArg, ComplementArg
from pgse.pipeline.pgse_inference_pipeline import Pipeline as InferencePipeline


class Pipeline(InferencePipeline):
    def __init__(
            self,
            model_path: str,
            segment_path: str,
            workers: int = 8,
            alphabet: AlphabetArg = None,
            case_sensitive: bool = False,
            complement: ComplementArg = AUTO
    ) -> None:
        super().__init__(
            model_path, segment_path, workers,
            alphabet=alphabet, case_sensitive=case_sensitive, complement=complement
        )

    def run(self, files: list[str]) -> np.ndarray:
        assert self.model is not None, 'The model failed to load.'

        loader = LoaderInference(files)
        data = loader.get_dataset_from_pool()

        dtest = xgb.DMatrix(data)
        preds = self.model.predict(dtest)

        return preds
