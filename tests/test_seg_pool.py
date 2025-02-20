import unittest

from pgse.segment.segment_pool import SegmentPool


class TestSegmentPool(unittest.TestCase):
    def setUp(self):
        self.segment_pool = SegmentPool()

    def test_redundant_elimination_empty(self):
        self.segment_pool.segments = []
        self.segment_pool.redundant_elimination([])
        self.assertEqual([], self.segment_pool.segments)

    def test_redundant_elimination_single_segment(self):
        self.segment_pool.segments = ['segment']
        self.segment_pool.redundant_elimination([0])
        self.assertEqual(['segment'], self.segment_pool.segments)

    def test_redundant_elimination_multiple_identical_segments(self):
        self.segment_pool.segments = ['segment', 'segment']
        self.segment_pool.redundant_elimination([0, 1])
        self.assertEqual(['segment'], self.segment_pool.segments)

    def test_redundant_elimination_no_substring(self):
        self.segment_pool.segments = ['segment', 'test']
        self.segment_pool.redundant_elimination([0, 1])
        self.assertEqual(['segment', 'test'], self.segment_pool.segments)

    def test_redundant_elimination_substring(self):
        self.segment_pool.segments = ['segment', 'seg']
        self.segment_pool.redundant_elimination([0, 1])
        self.assertEqual(['segment'], self.segment_pool.segments)

    def test_redundant_elimination_both_substrings(self):
        self.segment_pool.segments = ['segment', 'seg', 'ment']
        self.segment_pool.redundant_elimination([0, 1, 2])
        self.assertEqual(['segment'], self.segment_pool.segments)

    def test_redundant_elimination_multiple_substrings_longer_important(self):
        self.segment_pool.segments = ['segment', 'seg', 'ment', 'segm']
        self.segment_pool.redundant_elimination([0, 1, 2, 3])
        self.assertEqual(['segment'], self.segment_pool.segments)

    def test_redundant_elimination_multiple_substrings_shorter_important(self):
        self.segment_pool.segments = ['segment', 'seg', 'ment', 'segm']
        self.segment_pool.redundant_elimination([1, 0, 2, 3])
        self.assertEqual(['seg', 'ment'], self.segment_pool.segments)

    def test_redundant_elimination_multiple_substrings_middle_important(self):
        self.segment_pool.segments = ['segment', 'seg', 'ment', 'segm']
        self.segment_pool.redundant_elimination([1, 2, 0, 3])
        self.assertEqual(['seg', 'ment'], self.segment_pool.segments)

    def test_n_gram_grafting_non(self):
        self.segment_pool.segments = ['segment', 'seg', 'ment', 'segm']
        self.segment_pool._n_gram_grafting(4)
        self.assertEqual({'segment', 'seg', 'ment', 'segm'}, set(self.segment_pool.segments))

    def test_n_gram_grafting_empty(self):
        self.segment_pool.segments = []
        self.segment_pool._n_gram_grafting(4)
        self.assertEqual([], self.segment_pool.segments)

    def test_n_gram_grafting_single(self):
        self.segment_pool.segments = ['segment']
        self.segment_pool._n_gram_grafting(4)
        self.assertEqual(['segment'], self.segment_pool.segments)

    def test_n_gram_grafting_correct(self):
        self.segment_pool.segments = ['segment', 'menta']
        self.segment_pool._n_gram_grafting(4)
        self.assertEqual({'segment', 'menta', 'segmenta'}, set(self.segment_pool.segments))

    def test_n_gram_grafting_partly_matching(self):
        self.segment_pool.segments = ['segment', 'menaa']
        self.segment_pool._n_gram_grafting(4)
        self.assertEqual({'segment', 'menaa'}, set(self.segment_pool.segments))

    def test_n_gram_grafting_two_matching_w_one_short(self):
        self.segment_pool.segments = ['segment', 'menta', 'segm']
        self.segment_pool._n_gram_grafting(4)
        self.assertEqual({'segment', 'menta', 'segm', 'segmenta'}, set(self.segment_pool.segments))

    def test_n_gram_grafting_two_matching_w_one_long(self):
        self.segment_pool.segments = ['segment', 'menta', 'segmen']
        self.segment_pool._n_gram_grafting(4)
        self.assertEqual({'segment', 'menta', 'segmen', 'segmenta'}, set(self.segment_pool.segments))
        self.segment_pool._n_gram_grafting(3)
        self.assertEqual({'segment', 'menta', 'segmen', 'segmenta'}, set(self.segment_pool.segments))

    def test_n_gram_grafting_two_matching(self):
        self.segment_pool.segments = ['segment', 'menta', 'abcsegm']
        self.segment_pool._n_gram_grafting(4)
        self.assertEqual({'segment', 'menta', 'abcsegm', 'segmenta', 'abcsegment'}, set(self.segment_pool.segments))


if __name__ == '__main__':
    unittest.main()
