import glob
import json
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
from pandas import DataFrame, read_csv
from xgboost import Booster

from pgse import PGSEModel, TrainingPipeline
from pgse.result.segment_importance import SegmentImportance


def read_labels(label_file: str, genomes: list) -> np.ndarray:
    """Return the label of each genome, in the order the genomes are given.

    Args:
        label_file: Path of the CSV holding a labels and a files column.
        genomes: Paths of the genome files to look up.
    """
    labels = read_csv(label_file).set_index('files')['labels']
    return np.array([labels[os.path.basename(path)] for path in genomes])


class TestTrainingPipeline(unittest.TestCase):
    def test_training_pipeline(self):
        with TemporaryDirectory() as tmp_dir:
            pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                        label_file="resource/labels.csv",
                                        label_columns="labels",
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
                                        label_columns="labels",
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
                                        label_file="resource/labels.csv",
                                        label_columns="labels")
            pipeline.run()

            self.assertEqual(before, set(glob.glob(os.path.join(os.getcwd(), '*'))))
            self.assertEqual([], os.listdir(tmp_dir))

    def test_functional_pipeline(self):
        pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                    label_file="resource/labels.csv",
                                    label_columns="labels",
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
                                              label_columns="labels",
                                              folds=2)
        results_abs_paths = pipeline_abs_paths.train()
        self.assertEqual(results.segments.segments, results_abs_paths.segments.segments)
        for fold, fold_abs in zip(results.folds, results_abs_paths.folds):
            self.assertEqual(list(fold.predictions['Prediction']), list(fold_abs.predictions['Prediction']))
            self.assertEqual(list(fold.predictions['Actual']), list(fold_abs.predictions['Actual']))


class TestBinaryTrainingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.genomes = sorted(glob.glob('resource/genomes_binary/*.fna'))
        cls.result = TrainingPipeline(data_dir="resource/genomes_binary/",
                                      label_file="resource/labels_binary.csv",
                                      label_columns="labels",
                                      k=4, target=14, binary=True).train()
        cls.model = cls.result.model

    def test_the_run_defaults_to_auroc(self):
        self.assertEqual('auroc', self.result.metric)
        self.assertTrue(self.result.greater_is_better)

    def test_the_planted_motif_separates_the_classes(self):
        self.assertGreaterEqual(self.result.score, 0.9)

    def test_the_booster_is_a_logistic_classifier(self):
        config = json.loads(self.model.booster.save_config())
        self.assertEqual('binary:logistic', config['learner']['objective']['name'])

    def test_predictions_are_probabilities(self):
        predictions = self.model.predict(self.genomes)

        self.assertEqual(len(self.genomes), len(predictions))
        self.assertTrue(((predictions >= 0.0) & (predictions <= 1.0)).all())

    def test_the_positives_score_above_the_negatives(self):
        predictions = self.model.predict(self.genomes)
        labels = read_labels('resource/labels_binary.csv', self.genomes)

        self.assertGreater(predictions[labels == 1].min(), predictions[labels == 0].max())

    def test_binary_mode_survives_a_save_and_load(self):
        with TemporaryDirectory() as tmp_dir:
            self.model.save(os.path.join(tmp_dir, 'model'))
            loaded = PGSEModel.load(os.path.join(tmp_dir, 'model'))

            self.assertTrue(loaded.binary)
            np.testing.assert_allclose(
                self.model.predict(self.genomes),
                loaded.predict(self.genomes)
            )

    def test_boolean_labels_train_the_same_model(self):
        labels = read_labels('resource/labels_binary.csv', self.genomes)
        booleans = {path: bool(label) for path, label in zip(self.genomes, labels)}

        result = TrainingPipeline(data_dir="", label_file=booleans, label_columns="labels",
                                  k=4, target=14, binary=True).train()

        self.assertEqual('auroc', result.metric)
        np.testing.assert_allclose(
            self.model.predict(self.genomes),
            result.model.predict(self.genomes)
        )

    def test_a_non_binary_label_is_rejected(self):
        pipeline = TrainingPipeline(data_dir="resource/genomes/",
                                    label_file="resource/labels.csv",
                                    label_columns="labels",
                                    binary=True)

        with self.assertRaises(ValueError) as caught:
            pipeline.train()
        self.assertIn('0/1 labels', str(caught.exception))


class TestPGSEModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.genomes = sorted(glob.glob('resource/genomes/*.fna'))
        cls.model = TrainingPipeline(data_dir="resource/genomes/",
                                     label_file="resource/labels.csv",
                                     label_columns="labels").train().model

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

    def test_a_regression_model_records_that_it_is_not_binary(self):
        self.assertFalse(self.model.binary)

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
