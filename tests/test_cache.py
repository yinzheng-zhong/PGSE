import unittest

from pgse.genome.cache import Cache


class TestCache(unittest.TestCase):

    def setUp(self):
        self.cache_class = Cache(num_nodes=3)

    def test_get_existing_segment(self):
        self.cache_class.set('segment1', 5, 0)
        result = self.cache_class.get('segment1', 0)
        self.assertIsNotNone(result)
        self.assertIn('count', result)
        self.assertIn('indices', result)
        self.assertIsInstance(result['count'], int)
        self.assertIsInstance(result['indices'], list)
        result_none = self.cache_class.get('segment1', 1)
        self.assertIsNone(result_none)

    def test_get_non_existing_segment(self):
        result = self.cache_class.get('segment2', 0)
        self.assertIsNone(result)

    def test_set_two_indices(self):
        self.cache_class.set('segment3', 1, 0)
        self.cache_class.set('segment3', 2, 0)
        result = self.cache_class.get('segment3', 0)
        self.assertEqual(result['count'], 2, "Count was not 2 after setting two indices")
        self.assertEqual(set(result['indices']), {1, 2}, "Indices were not [1, 2] after setting two indices")

    def test_refresh_with_no_segments(self):
        self.cache_class.refresh()
        self.assertEqual(
            sum(len(cache) for cache in self.cache_class._caches),
            0,
            "Cache was not empty after refresh with no segments"
        )

    def test_refresh_with_ttl_greater_than_zero(self):
        self.cache_class.set("segment1", 1, 0)
        self.cache_class._caches[0]["segment1"]['ttl'] = 1
        self.cache_class.refresh()
        self.assertIn(
            "segment1",
            self.cache_class._caches[0],
            "Cache did not keep segments with ttl greater than 0 after refresh"
        )

    def test_refresh_with_ttl_zero(self):
        self.cache_class.set("segment2", 2, 0)
        self.cache_class._caches[0]["segment2"]['ttl'] = 0
        self.cache_class.refresh()
        self.assertNotIn(
            "segment2",
            self.cache_class._caches[0],
            "Cache did not remove segments with ttl of 0 after refresh"
        )

    def test_refresh_with_ttl_less_than_zero(self):
        self.cache_class.set("segment3", 3, 0)
        self.cache_class._caches[0]["segment3"]['ttl'] = -1
        self.cache_class.refresh()
        self.assertNotIn(
            "segment3",
            self.cache_class._caches[0],
            "Cache did not remove segments with ttl less than 0 after refresh"
        )
