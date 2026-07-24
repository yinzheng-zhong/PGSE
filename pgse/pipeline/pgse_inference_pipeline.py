from typing import Optional

import ray
from pgse.environment.ray_env import RayEnvManager
from pgse.dataset.loader_inference import LoaderInference
import numpy as np
import xgboost as xgb

from pgse.etc.alphabet import AUTO, Alphabet, AlphabetArg, ComplementArg, set_alphabet
from pgse.log import logger
from pgse.segment import seg_pool


class Pipeline:
    def __init__(
            self,
            model_path: str,
            segment_path: str,
            workers: int = 8,
            alphabet: AlphabetArg = None,
            case_sensitive: bool = False,
            complement: ComplementArg = AUTO
    ) -> None:
        """
        :param alphabet: str or Alphabet: The alphabet the model was trained with.
            It must match, otherwise the segments cannot be counted.
        :param case_sensitive: bool: Treat upper and lower case as distinct characters.
        :param complement: The complement used to canonicalise segments.
        """
        self.model_path: str = model_path
        self.segment_path: str = segment_path
        self.alphabet: Alphabet = set_alphabet(alphabet, case_sensitive=case_sensitive, complement=complement)
        logger.info(f'Using {self.alphabet}')

        RayEnvManager.initialize(False, 0, workers)

        self.model_params: dict = {
            'nthread': workers,
        }
        self.model: Optional[xgb.Booster] = None
        self._load()

    def _load(self) -> None:
        self.model = xgb.Booster(params=self.model_params, model_file=self.model_path)
        seg_pool.import_segments(self.segment_path)
        self._check_segments_match_alphabet()

    def _check_segments_match_alphabet(self) -> None:
        """
        The segments file does not record the alphabet it was built with, so a mismatch
        would silently produce all-zero counts. Fail loudly instead.
        """
        allowed = set(self.alphabet.characters(self.alphabet.unknown_char is not None))
        unexpected = {
            char
            for segment in seg_pool
            for char in self.alphabet.normalise(segment)
            if char not in allowed
        }

        if unexpected:
            raise ValueError(
                f'The segments in {self.segment_path} contain characters outside {self.alphabet}: '
                f'{sorted(unexpected)}. Pass the alphabet the model was trained with.'
            )

    def run(self, files: list[str]) -> np.ndarray:
        assert self.model is not None, 'The model failed to load.'

        loader = LoaderInference(files)
        data = loader.get_dataset_from_pool()

        dtest = xgb.DMatrix(data)
        preds = self.model.predict(dtest)

        ray.shutdown()

        return preds
