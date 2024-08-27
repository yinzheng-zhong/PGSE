import unittest

from src.genome.cache import Cache


class TestCache(unittest.TestCase):

    def setUp(self):
        self.cache = Cache()

    def test_get_existing_segment(self):
        self.cache.set('segment1', 5)
        result = self.cache.get('segment1')
        self.assertIsNotNone(result)
        self.assertIn('count', result)
        self.assertIn('indices', result)
        self.assertIsInstance(result['count'], int)
        self.assertIsInstance(result['indices'], list)

    def test_get_non_existing_segment(self):
        result = self.cache.get('segment2')
        self.assertIsNone(result)

    def test_set_two_indices(self):
        self.cache.set('segment3', 1)
        self.cache.set('segment3', 2)
        result = self.cache.get('segment3')
        self.assertEqual(result['count'], 2, "Count was not 2 after setting two indices")
        self.assertEqual(set(result['indices']), {1, 2}, "Indices were not [1, 2] after setting two indices")

    def test_refresh_with_no_segments(self):
        self.cache.refresh()
        self.assertEqual(len(self.cache.cache), 0, "Cache was not empty after refresh with no segments")

    def test_refresh_with_ttl_greater_than_zero(self):
        self.cache.set("segment1", 1)
        self.cache.cache["segment1"]['ttl'] = 1
        self.cache.refresh()
        self.assertIn("segment1", self.cache.cache, "Cache did not keep segments with ttl greater than 0 after refresh")

    def test_refresh_with_ttl_zero(self):
        self.cache.set("segment2", 2)
        self.cache.cache["segment2"]['ttl'] = 0
        self.cache.refresh()
        self.assertNotIn("segment2", self.cache.cache, "Cache did not remove segments with ttl of 0 after refresh")

    def test_refresh_with_ttl_less_than_zero(self):
        self.cache.set("segment3", 3)
        self.cache.cache["segment3"]['ttl'] = -1
        self.cache.refresh()
        self.assertNotIn("segment3", self.cache.cache, "Cache did not remove segments with ttl less than 0 after refresh")
