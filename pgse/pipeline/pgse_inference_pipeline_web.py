from pgse.dataset.loader_inference import LoaderInference
import xgboost as xgb
from pgse.pipeline.pgse_inference_pipeline import Pipeline as InferencePipeline


class Pipeline(InferencePipeline):
    def __init__(self, model_path: str, segment_path: str, workers: int = 8):
        super().__init__(model_path, segment_path, workers)

    def run(self, files: [str]):
        loader = LoaderInference(files)
        data = loader.get_dataset_from_pool()

        dtest = xgb.DMatrix(data)
        preds = self.model.predict(dtest)

        return preds
