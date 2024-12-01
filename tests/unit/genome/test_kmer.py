import unittest


from src.genome import canonicalize
from src.genome.kmer import Kmer


class TestUtil(unittest.TestCase):

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
                result = canonicalize(sequence)
                self.assertEqual(result, expected)

    def test_kmer_mapping(self):
        test_cases = [
            ("aaaaaagg", 10),
            ('gaatgcag', 43),
            ('aaaaggct', 2),
        ]

        for sequence, expected in test_cases:
            with self.subTest(sequence=sequence):
                result = self.kmer.kmer_mapping(sequence)
                self.assertEqual(result, sequence)


if __name__ == '__main__':
    unittest.main()
