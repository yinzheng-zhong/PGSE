"""The predictive model a PGSE run produces."""

import json
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from pgse.dataset.alphabet import Alphabet
from pgse.dataset.alphabet_utils import alphabet_from_dict, alphabet_to_dict, using_alphabet
from pgse.dataset.counts import Dataset
from pgse.genome.sequence import Sequence
from pgse.log import logger
from pgse.model.segment_counter import SegmentCounter
from pgse.result.segment_importance import SegmentImportance


class PGSEModel:
    """A trained model with the segments it reads and everything needed to count them.

    Predicting touches no files and no global state beyond the alphabet, which is
    restored afterwards, so several models can be held and used at once.
    """

    MODEL_SUFFIX = '.json'
    SEGMENTS_SUFFIX = '_segs.csv'
    METADATA_SUFFIX = '_meta.json'

    def __init__(
            self,
            booster: xgb.Booster,
            segments: SegmentImportance,
            alphabet: Alphabet,
            count_dtype: npt.DTypeLike = np.float32,
            sparse: bool = False,
            workers: int = 8
    ) -> None:
        """
        Args:
            booster: The trained XGBoost model.
            segments: The segments the booster was trained on, in column order.
            alphabet: The alphabet the segments were counted with.
            count_dtype: Storage dtype of the count matrix (np.float32 or np.uint16).
            sparse: Store the count matrix as a sparse CSR matrix.
            workers: Threads used for counting and prediction.
        """
        self.booster: xgb.Booster = booster
        self.segments: SegmentImportance = segments
        self.alphabet: Alphabet = alphabet
        self.count_dtype: npt.DTypeLike = count_dtype
        self.sparse: bool = sparse
        self.workers: int = workers

        self.counter: SegmentCounter = SegmentCounter(
            segments.segments, alphabet, count_dtype=count_dtype, sparse=sparse, threads=workers
        )

    def __repr__(self) -> str:
        return f'PGSEModel({len(self.segments)} segments, {self.alphabet})'

    def predict(self, files: list[str]) -> np.ndarray:
        """Predict a value for each sequence file.

        Args:
            files: Paths of the FASTA files to score.
        """
        return self._predict(self._read(files=files))

    def predict_sequences(self, sequences: list[str]) -> np.ndarray:
        """Predict a value for each in-memory sequence.

        Args:
            sequences: The sequences themselves, either FASTA text or bare sequences.
        """
        return self._predict(self._read(texts=sequences))

    def count(
            self,
            files: Optional[list[str]] = None,
            sequences: Optional[list[str]] = None
    ) -> Dataset:
        """Build the segment-count matrix of the input, one row per sequence.

        Args:
            files: Paths of the FASTA files to count.
            sequences: In-memory sequences to count, used when files is not given.
        """
        return self.counter.count(self._read(files=files, texts=sequences))

    def metadata(self) -> dict[str, Any]:
        """Everything save writes besides the booster and the segments."""
        try:
            pgse_version = version('pgse')
        except PackageNotFoundError:
            pgse_version = 'unknown'

        return {
            'pgse_version': pgse_version,
            'alphabet': alphabet_to_dict(self.alphabet),
            'count_dtype': np.dtype(self.count_dtype).name,
            'sparse': self.sparse,
        }

    def save(self, path: str) -> list[str]:
        """Write the booster, the segments and the metadata, all sharing one path prefix.

        Args:
            path: Prefix of the files to write, e.g. 'out/model' writes out/model.json,
                out/model_segs.csv and out/model_meta.json.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        model_path, segments_path, metadata_path = self._paths(path)

        self.booster.save_model(model_path)
        self.segments.to_csv(segments_path)
        with open(metadata_path, 'w') as file:
            json.dump(self.metadata(), file, indent=2)

        logger.info(f'Saved the model to {model_path}, {segments_path} and {metadata_path}')
        return [model_path, segments_path, metadata_path]

    @classmethod
    def load(cls, path: str, workers: int = 8) -> 'PGSEModel':
        """Load a model written by save.

        Args:
            path: The prefix that was passed to save.
            workers: Threads used for counting and prediction.
        """
        model_path, segments_path, metadata_path = cls._paths(path)

        booster = xgb.Booster(params={'nthread': workers}, model_file=model_path)
        segments = SegmentImportance.from_csv(segments_path)

        if os.path.exists(metadata_path):
            with open(metadata_path) as file:
                metadata = json.load(file)
            alphabet = alphabet_from_dict(metadata['alphabet'])
            count_dtype = np.dtype(metadata.get('count_dtype', 'float32')).type
            sparse = bool(metadata.get('sparse', False))
        else:
            logger.warning(
                f'{metadata_path} is missing, so the default DNA alphabet and dense float32 '
                f'counts are assumed. Rebuild the model with save to record them.'
            )
            alphabet, count_dtype, sparse = Alphabet(), np.float32, False

        return cls(booster, segments, alphabet, count_dtype=count_dtype, sparse=sparse, workers=workers)

    @classmethod
    def _paths(cls, path: str) -> tuple[str, str, str]:
        """The booster, segment and metadata paths for a prefix.

        Args:
            path: The prefix the three files share.
        """
        return path + cls.MODEL_SUFFIX, path + cls.SEGMENTS_SUFFIX, path + cls.METADATA_SUFFIX

    def _read(
            self,
            files: Optional[list[str]] = None,
            texts: Optional[list[str]] = None
    ) -> list[Sequence]:
        """Read the input into Sequence objects under the model's own alphabet.

        Args:
            files: Paths of the FASTA files to read.
            texts: In-memory sequences to read, used when files is not given.
        """
        if files is None and texts is None:
            raise ValueError('Pass either files or sequences.')

        with using_alphabet(self.alphabet):
            if files is not None:
                return [Sequence(file) for file in files]
            return [Sequence(text=text) for text in texts or []]

    def _predict(self, sequences: list[Sequence]) -> np.ndarray:
        """Count the sequences and run them through the booster.

        Args:
            sequences: The sequences to score.
        """
        counts = self.counter.count(sequences)
        return self.booster.predict(xgb.DMatrix(counts, nthread=self.workers))
