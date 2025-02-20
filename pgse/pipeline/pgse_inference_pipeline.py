from pgse.dataset.loader_inference import LoaderInference
import xgboost as xgb

from pgse.segment import seg_pool


class Pipeline:
    def __init__(self, model_path: str, segment_path: str):
        self.model_path = model_path
        self.segment_path = segment_path

        self.model = None
        self._load()

    def _load(self):
        self.model = xgb.Booster(model_file=self.model_path)
        seg_pool.import_segments(self.segment_path)

    def run(self, files: [str]):
        loader = LoaderInference(files)
        data = loader.get_dataset_from_pool()

        dtest = xgb.DMatrix(data)
        preds = self.model.predict(dtest)

        return preds
