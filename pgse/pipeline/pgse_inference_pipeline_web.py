from typing import Optional

import numpy as np
import xgboost as xgb
from pgse.dataset.alphabet import AUTO, AlphabetArg, ComplementArg
from pgse.pipeline.pgse_inference_pipeline import Pipeline as InferencePipeline


class Pipeline(InferencePipeline):
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
        super().__init__(
            model_path, segment_path, workers,
            alphabet=alphabet, case_sensitive=case_sensitive, complement=complement,
            uint16=uint16, sparse=sparse
        )

    def run(self, files: Optional[list[str]] = None, sequences: Optional[list[str]] = None) -> np.ndarray:
        """Score every input, keeping Ray alive for the next call.

        Args:
            files: Paths of the FASTA files to score.
            sequences: In-memory sequences to score, used when files is not given.
        """
        assert self.model is not None, 'The model failed to load.'

        data = self._count(files, sequences)

        dtest = xgb.DMatrix(data)
        preds = self._in_label_units(self.model.predict(dtest))

        return preds
