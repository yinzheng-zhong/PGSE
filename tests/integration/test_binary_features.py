import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from pgse import PGSEModel, TrainingPipeline
from pgse.dataset.alphabet import reset_alphabet

MOTIF = 'aaggttcc'


def samples(count: int = 60) -> pd.DataFrame:
    """Build a table of sequences whose label is driven by whether a motif occurs.

    Args:
        count: Number of samples to build.
    """
    rng = np.random.default_rng(11)
    rows = []

    for index in range(count):
        sequence = ''.join(rng.choice(list('atgc'), size=120))
        has_motif = bool(index % 2)
        if has_motif:
            sequence = sequence[:30] + MOTIF + sequence[30:]
        rows.append({'sequence': sequence, 'mic': 4.0 if has_motif else 0.0})

    return pd.DataFrame(rows)


class TestBinaryFeatures(unittest.TestCase):
    """A pipeline run with binary_features trains, predicts and saves on 0/1 presence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = samples()
        cls.result = TrainingPipeline(
            table_file=cls.table,
            data_column='sequence',
            label_columns='mic',
            k=4,
            ext=2,
            target=8,
            num_rounds=40,
            metric='rmse',
            workers=2,
            binary_features=True,
        ).train()
        cls.model = cls.result.model

    @classmethod
    def tearDownClass(cls) -> None:
        reset_alphabet()

    def test_features_are_zero_or_one(self) -> None:
        repeated = MOTIF * 3 + ''.join(['ac'] * 20)

        features = self.model.count(sequences=self.table['sequence'].tolist() + [repeated])

        self.assertTrue(set(np.unique(features)) <= {0.0, 1.0})
        self.assertEqual(1.0, float(features.max()))

    def test_the_setting_is_saved_with_the_model(self) -> None:
        with TemporaryDirectory() as directory:
            prefix = os.path.join(directory, 'model')
            self.model.save(prefix)

            reloaded = PGSEModel.load(prefix, workers=2)

        self.assertTrue(self.model.binary_features)
        self.assertTrue(reloaded.binary_features)
        np.testing.assert_allclose(
            self.model.predict_sequences(self.table['sequence'].tolist()),
            reloaded.predict_sequences(self.table['sequence'].tolist()),
            rtol=1e-6,
        )


if __name__ == '__main__':
    unittest.main()
