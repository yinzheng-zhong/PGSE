import unittest
import os
from tempfile import TemporaryDirectory
from xgboost import Booster
from pandas import DataFrame

from pgse import TrainingPipeline


class TestTrainingPipeline(unittest.TestCase):
    def test_training_pipeline(self):
        with TemporaryDirectory() as tmp_dir:
            pipeline = TrainingPipeline(data_dir = "resource/genomes/",
                                        label_file = "resource/labels.csv",
                                        save_file=os.path.join(tmp_dir, "save"),
                                        export_file=os.path.join(tmp_dir, "export"))
            pipeline.run()

    def test_no_write_pipeline(self):
        with TemporaryDirectory() as tmp_dir:
            pipeline = TrainingPipeline(data_dir = "resource/genomes/",
                                        label_file = "resource/labels.csv",
                                        save_file=os.path.join(tmp_dir, "save"),
                                        export_file=os.path.join(tmp_dir, "export"))
            pipeline._suppress_write = True
            pipeline.run()
            # list files in tmp_dir
            files = os.listdir(tmp_dir)
            self.assertEqual(len(files), 0)

    def test_functional_pipeline(self):
        pipeline = TrainingPipeline(data_dir = "resource/genomes/",
                                    label_file = "resource/labels.csv")
        results = pipeline.train()

        self.assertIsInstance(results.segments, list)
        self.assertIsInstance(results.models, list)
        self.assertIsInstance(results.results, list)

        for model in results.models:
            self.assertIsInstance(model, Booster)
        for segment in results.segments:
            self.assertIsInstance(segment, list)
        for result in results.results:
            self.assertIsInstance(result, DataFrame)
            self.assertIn('Prediction', result)
            self.assertIn('Actual', result)
