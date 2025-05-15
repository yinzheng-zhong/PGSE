import ray
from pgse.environment.ray_env import RayEnvManager
from pgse.dataset.loader_inference import LoaderInference
import xgboost as xgb

from pgse.segment import seg_pool


class Pipeline:
    def __init__(
            self,
            model_path: str,
            segment_path: str,
            workers: int = 8,
    ):
        self.model_path = model_path
        self.segment_path = segment_path
        RayEnvManager.initialize(False, 0, workers)

        self.model_params = {
            'nthread': workers,
        }
        self.model = None
        self._load()

    def _load(self):
        self.model = xgb.Booster(params=self.model_params, model_file=self.model_path)
        seg_pool.import_segments(self.segment_path)

    def run(self, files: list[str]):
        loader = LoaderInference(files)
        data = loader.get_dataset_from_pool()

        dtest = xgb.DMatrix(data)
        preds = self.model.predict(dtest)

        ray.shutdown()

        return preds
