import unittest
from src.genome.kmer import Kmer


class TestKmer(unittest.TestCase):

    def setUp(self):
        self.kmer = Kmer()

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
                result = self.kmer.convert_to_canonical(sequence)
                self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
