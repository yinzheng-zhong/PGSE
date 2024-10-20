import unittest
from unittest.mock import MagicMock

from src.genome.sequence import Sequence
from time import time
from src.segment import seg_pool


class TestSequence(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence('test_sequence.txt')
        seg_pool.segments = ['aactgccaggcatcaaatt', 'aactgccaggcatcaaat']

    def test_read_sequence(self):
        self.assertEqual('aactgccaggcatcaaattagat', str(self.sequence))

    def test_get_kmer_count_with_consecutive(self):
        count = self.sequence.get_kmer_count(2, no_consecutive=False)
        self.assertEqual(
            [4, 2, 3, 1, 1, 0, 0, 0, 2, 0, 0, 2, 4, 0, 0, 2],
            list(count)
        )

    def test_get_kmer_count_with_2_nodes(self):
        self.sequence._nodes[1] = self.sequence._nodes[0]
        count = self.sequence.get_kmer_count(2, no_consecutive=False)
        self.assertEqual(
            [8, 4, 6, 2, 2, 0, 0, 0, 4, 0, 0, 4, 8, 0, 0, 4],
            list(count)
        )

    def test_get_kmer_count_2_nodes_different_length(self):
        self.sequence._nodes[1] = 'aa'
        count = self.sequence.get_kmer_count(2, no_consecutive=False)
        self.assertEqual(
            [5, 2, 3, 1, 1, 0, 0, 0, 2, 0, 0, 2, 4, 0, 0, 2],
            list(count)
        )

    def test_occurrence(self):
        string = 'aactgccaggcatcaaattagat'
        count_overlapping = self.sequence._occurrences_overlapping(string, 'aa')
        self.assertEqual(3, count_overlapping)

        count_non_overlapping = self.sequence._occurrences(string, 'aa')
        self.assertEqual(2, count_non_overlapping)

    def test_get_count_from_seg_manager(self):
        self.sequence._nodes[0] = 'aactgccaggcatcaaattt' * 100
        count = self.sequence.get_count_from_seg_manager(seg_pool)
        self.assertEqual(
            [100, 100],
            list(count)
        )

        self.sequence._nodes[1] = 'aactgccaggcatcaaattt'
        count = self.sequence.get_count_from_seg_manager(seg_pool)
        self.assertEqual(
            [101, 101],
            list(count)
        )
