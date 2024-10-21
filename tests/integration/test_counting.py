import unittest

from src.genome.sequence import Sequence
from src.segment import seg_pool


class TestSequence(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence('../resource/test_sequence.txt')

    def test_get_count_from_seg_manager(self):
        seg_pool.segments = ['aactgccaggcatcaaatt', 'aactgccaggcatcaaat']

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

    def test_kmer_count_match_seg_pool_count(self):
        # count kmer from the sequence first
        kmer_count = self.sequence.get_kmer_count(2, no_consecutive=False)
        seg_pool.add_all_kmer(2, 2)

        seg_pool_count = self.sequence.get_count_from_seg_manager(seg_pool)

        self.assertEqual(
            list(kmer_count),
            list(seg_pool_count)
        )
