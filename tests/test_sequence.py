import unittest

from src.genome.sequence import Sequence


class TestSequence(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence('test_sequence.txt')

    def test_read_sequence(self):
        self.assertEqual('aactgccaggcatcaaattagat', str(self.sequence))

    def test_get_kmer_count_with_consecutive(self):
        count = self.sequence.get_kmer_count(2, no_consecutive=False)
        self.assertEqual(
            [4, 3, 3, 1, 1, 0, 0, 0, 2, 0, 0, 2, 4, 0, 0, 2],
            list(count)
        )

    def test_occurrence(self):
        string = 'aactgccaggcatcaaattagat'
        count_overlapping = self.sequence._occurrences_overlapping(string, 'aa')
        self.assertEqual(3, count_overlapping)

        count_non_overlapping = self.sequence._occurrences(string, 'aa')
        self.assertEqual(2, count_non_overlapping)
