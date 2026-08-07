import glob
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
from pandas import DataFrame
from xgboost import Booster

from pgse import PGSEModel, TrainingPipeline
from pgse.result.segment_importance import SegmentImportance


class TestTrainingPipeline(unittest.TestCase):
    def test_training_pipeline(self):
        with TemporaryDirectory() as tmp_dir:
            pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                        label_file="resource/labels.csv",
                                        save_file=os.path.join(tmp_dir, "save"),
                                        export_file=os.path.join(tmp_dir, "export"))
            pipeline.run()

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "export_fold_0.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "export_fold_0_segs.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "export_fold_0_meta.json")))

    def test_no_write_pipeline(self):
        with TemporaryDirectory() as tmp_dir:
            pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                        label_file="resource/labels.csv",
                                        save_file=os.path.join(tmp_dir, "save"),
                                        export_file=os.path.join(tmp_dir, "export"))
            pipeline.train()
            # list files in tmp_dir
            files = os.listdir(tmp_dir)
            self.assertEqual(len(files), 0)

    def test_nothing_is_written_without_paths(self):
        with TemporaryDirectory() as tmp_dir:
            before = set(glob.glob(os.path.join(os.getcwd(), '*')))
            pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                        label_file="resource/labels.csv")
            pipeline.run()

            self.assertEqual(before, set(glob.glob(os.path.join(os.getcwd(), '*'))))
            self.assertEqual([], os.listdir(tmp_dir))

    def test_functional_pipeline(self):
        pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                    label_file="resource/labels.csv",
                                    folds=2)
        results = pipeline.train()

        self.assertEqual(2, len(results))
        self.assertIsInstance(results.predictions, DataFrame)
        self.assertIsInstance(results.segments, SegmentImportance)
        self.assertIsInstance(results.model, PGSEModel)
        self.assertIsInstance(results.model.booster, Booster)
        self.assertIn(results.model, results.models)

        for fold in results.folds:
            self.assertIsInstance(fold.model, PGSEModel)
            self.assertIsInstance(fold.segments, SegmentImportance)
            self.assertIsInstance(fold.score, float)
            self.assertIsInstance(fold.predictions, DataFrame)
            self.assertIn('Prediction', fold.predictions)
            self.assertIn('Actual', fold.predictions)
            self.assertEqual(len(fold.segments), len(fold.model.segments.segments))

        # test that absolute paths to .fna files give the same results
        pipeline_abs_paths = TrainingPipeline(data_dir="",
                                    label_file="resource/labels_full_paths.csv",
                                              folds=2)
        results_abs_paths = pipeline_abs_paths.train()
        self.assertEqual(results.segments.segments, results_abs_paths.segments.segments)
        for fold, fold_abs in zip(results.folds, results_abs_paths.folds):
            self.assertEqual(list(fold.predictions['Prediction']), list(fold_abs.predictions['Prediction']))
            self.assertEqual(list(fold.predictions['Actual']), list(fold_abs.predictions['Actual']))


class TestPGSEModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.genomes = sorted(glob.glob('resource/genomes/*.fna'))
        cls.model = TrainingPipeline(data_dir="resource/genomes/",
                                     label_file="resource/labels.csv").train().model

    def test_predict_from_files(self):
        predictions = self.model.predict(self.genomes)

        self.assertEqual(len(self.genomes), len(predictions))
        self.assertTrue(np.isfinite(predictions).all())

    def test_predict_from_sequences_matches_files(self):
        texts = [open(path).read() for path in self.genomes]

        np.testing.assert_allclose(
            self.model.predict(self.genomes),
            self.model.predict_sequences(texts)
        )

    def test_counts_are_one_row_per_sequence(self):
        counts = self.model.count(files=self.genomes)

        self.assertEqual((len(self.genomes), len(self.model.segments)), counts.shape)

    def test_save_and_load_round_trip(self):
        with TemporaryDirectory() as tmp_dir:
            written = self.model.save(os.path.join(tmp_dir, 'model'))
            for path in written:
                self.assertTrue(os.path.exists(path))

            loaded = PGSEModel.load(os.path.join(tmp_dir, 'model'))

            self.assertEqual(self.model.segments.segments, loaded.segments.segments)
            self.assertEqual(self.model.alphabet, loaded.alphabet)
            np.testing.assert_allclose(
                self.model.predict(self.genomes),
                loaded.predict(self.genomes)
            )

    def test_importance_is_aligned_with_the_segments(self):
        segments = self.model.segments

        self.assertEqual(len(segments.segments), len(segments.importances))
        self.assertGreater(np.count_nonzero(segments.importances), 0)

        best_segment, best_importance = segments.top(1)[0]
        self.assertEqual(best_importance, float(segments.importances.max()))
        self.assertEqual(best_segment, segments.segments[int(np.argmax(segments.importances))])
