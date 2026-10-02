import json
import os
from typing import Optional

from pgse.environment.ray_env import RayEnvManager
from pgse.dataset.loader_inference import LoaderInference
import numpy as np
import xgboost as xgb

from pgse.dataset.alphabet import AUTO, Alphabet, AlphabetArg, ComplementArg, set_alphabet
from pgse.log import logger
from pgse.model.label_scaler import LabelScaler
from pgse.model.pgse_model import PGSEModel
from pgse.segment import seg_pool


class Pipeline:
    def __init__(
            self,
            model_path: str,
            segment_path: str,
            workers: int = 8,
            alphabet: AlphabetArg = None,
            case_sensitive: bool = False,
            complement: ComplementArg = AUTO,
            uint16: bool = False,
            sparse: bool = False
    ) -> None:
        """
        :param alphabet: str or Alphabet: The alphabet the model was trained with.
            It must match, otherwise the segments cannot be counted.
        :param case_sensitive: bool: Treat upper and lower case as distinct characters.
        :param complement: The complement used to canonicalise segments.
        :param uint16: Store segment counts as uint16 instead of float32.
        :param sparse: Store the count matrix as sparse CSR. This must match the value
            used at train time: CSR's unstored zeros are read as missing by XGBoost,
            so mixing dense training with sparse prediction (or vice versa) shifts the
            predictions.
        """
        self.model_path: str = model_path
        self.segment_path: str = segment_path
        self.count_dtype = np.uint16 if uint16 else np.float32
        self.sparse: bool = sparse
        self.workers: int = workers
        self.alphabet: Alphabet = set_alphabet(alphabet, case_sensitive=case_sensitive, complement=complement)
        logger.info(f'Using {self.alphabet}')

        RayEnvManager.initialize(False, 0, workers)

        self.model_params: dict = {
            'nthread': workers,
        }
        self.model: Optional[xgb.Booster] = None
        # Read from the model's metadata file, when it was saved next to the model.
        self.label_names: list[str] = []
        self.scaler: Optional[LabelScaler] = None
        self._load()

    def _load(self) -> None:
        self.model = xgb.Booster(params=self.model_params, model_file=self.model_path)
        seg_pool.import_segments(self.segment_path)
        self._check_segments_match_alphabet()
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Read the label names and the label scaler from the model's metadata file."""
        if not self.model_path.endswith(PGSEModel.MODEL_SUFFIX):
            return

        prefix = self.model_path[:-len(PGSEModel.MODEL_SUFFIX)]
        metadata_path = prefix + PGSEModel.METADATA_SUFFIX
        if not os.path.exists(metadata_path):
            return

        with open(metadata_path) as file:
            metadata = json.load(file)

        self.label_names = metadata.get('label_names') or []
        if metadata.get('label_scaler'):
            self.scaler = LabelScaler.from_dict(metadata['label_scaler'])
            logger.info(f'Undoing the label standardisation recorded in {metadata_path}')

    def _in_label_units(self, predictions: np.ndarray) -> np.ndarray:
        """Return the predictions in the units of the labels the model was trained on.

        Args:
            predictions: What the booster predicted.
        """
        return predictions if self.scaler is None else self.scaler.inverse_transform(predictions)

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

    def run(self, files: Optional[list[str]] = None, sequences: Optional[list[str]] = None) -> np.ndarray:
        """Score every input, one prediction per sequence.

        Args:
            files: Paths of the FASTA files to score.
            sequences: In-memory sequences to score, used when files is not given.
        """
        assert self.model is not None, 'The model failed to load.'

        data = self._count(files, sequences)

        dtest = xgb.DMatrix(data)
        preds = self._in_label_units(self.model.predict(dtest))

        RayEnvManager.shutdown()

        return preds

    def _count(self, files: Optional[list[str]], sequences: Optional[list[str]]):
        """Build the segment-count matrix of the input, one row per sequence.

        Args:
            files: Paths of the FASTA files to count.
            sequences: In-memory sequences to count, used when files is not given.
        """
        if files is None and sequences is None:
            raise ValueError('Pass either files or sequences.')

        loader = LoaderInference(
            files if files is not None else sequences or [],
            inline=files is None,
            count_dtype=self.count_dtype,
            sparse=self.sparse,
            workers=self.workers
        )
        return loader.get_dataset_from_pool()
