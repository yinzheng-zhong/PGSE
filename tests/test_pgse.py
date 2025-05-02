import unittest
import os
from tempfile import TemporaryDirectory, TemporaryFile, NamedTemporaryFile
from xgboost import Booster
from pandas import DataFrame, read_csv

from pgse import TrainingPipeline


class TestTrainingPipeline(unittest.TestCase):
    def test_training_pipeline(self):
        with TemporaryDirectory() as tmp_dir:
            pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                        label_file="resource/labels.csv",
                                        save_file=os.path.join(tmp_dir, "save"),
                                        export_file=os.path.join(tmp_dir, "export"))
            pipeline.run()

    def test_no_write_pipeline(self):
        with TemporaryDirectory() as tmp_dir:
            pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                        label_file="resource/labels.csv",
                                        save_file=os.path.join(tmp_dir, "save"),
                                        export_file=os.path.join(tmp_dir, "export"))
            pipeline._suppress_write = True
            pipeline.run()
            # list files in tmp_dir
            files = os.listdir(tmp_dir)
            self.assertEqual(len(files), 0)

    def test_functional_pipeline(self):
        pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                    label_file="resource/labels.csv",
                                    folds=2)
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

        # test that absolute paths to .fna files give the same results
        pipeline_abs_paths = TrainingPipeline(data_dir="",
                                    label_file="resource/labels_full_paths.csv",
                                              folds=2)
        results_abs_paths = pipeline_abs_paths.train()
        self.assertEqual(results.segments, results_abs_paths.segments)
        for r, r_abs in zip(results.results, results_abs_paths.results):
            self.assertEqual(list(r['Prediction']), list(r_abs['Prediction']))
            self.assertEqual(list(r['Actual']), list(r_abs['Actual']))
