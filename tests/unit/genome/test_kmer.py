import unittest
from src.genome import canonicalize


class TestUtil(unittest.TestCase):

    def setUp(self):
        pass

    def test_canonical_reverse_complement(self):
        # Test cases
        test_cases = [
            ("atgc", "atgc"),  # same sequence
            ("gcat", "atgc"),  # reverse complement is smaller
            ("aaaa", "aaaa"),  # same sequence
            ("tttt", "aaaa"),  # reverse complement is smaller
            ("ta", "ta"),  # same sequence
        ]

        for sequence, expected in test_cases:
            with self.subTest(sequence=sequence):
                result = canonicalize(sequence)
                self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
