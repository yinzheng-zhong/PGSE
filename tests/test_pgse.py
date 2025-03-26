import unittest

from pgse import TrainingPipeline


class TestTrainingPipeline(unittest.TestCase):
    def test_training_pipeline(self):
        pipeline = TrainingPipeline(data_dir = "resource/genomes/", label_file = "resource/labels.csv")
        pipeline.run()
