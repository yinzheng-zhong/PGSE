import unittest

from pgse.algos import aho_corasick
from pgse.algos.aho_corasick_py import AhoCorasickPy
from pgse.dataset.alphabet import Alphabet, reset_alphabet, set_alphabet


class TestCountingWithAlphabets(unittest.TestCase):
    """
    Segment counting must give the same answer whichever implementation is active,
    so every case is checked against both the shared library (when it is compiled)
    and the pure Python fallback.
    """

    def setUp(self):
        self.implementations = [AhoCorasickPy()]
        if not isinstance(aho_corasick, AhoCorasickPy):
            self.implementations.append(aho_corasick)

    def tearDown(self):
        reset_alphabet()

    def assert_counts(self, nodes, segments, expected):
        for implementation in self.implementations:
            with self.subTest(implementation=type(implementation).__name__):
                self.assertEqual(expected, list(implementation.count_segments(nodes, segments)))

    def test_dna_counts_both_strands(self):
        # 'aactg' appears once, and once more as the reverse complement 'cagtt'
        self.assert_counts(['aactgccaggcagtt'], ['aactg'], [2])

    def test_dna_ignores_case(self):
        self.assert_counts(['AACTGCCAGG'], ['aactg'], [1])

    def test_dna_skips_non_canonical_segments(self):
        # 'cagtt' is the non-canonical form of 'aactg' and is never counted
        self.assert_counts(['aactgccagg'], ['cagtt'], [0])

    def test_text_alphabet_counts_every_segment_as_given(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        self.assert_counts(['the cat sat on the mat'], ['the', 'cat', 'at', 'zebra'], [2, 1, 3, 0])

    def test_text_alphabet_does_not_canonicalise(self):
        # 'eht' is 'the' reversed. Without a complement it must be counted separately.
        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        self.assert_counts(['the cat'], ['the', 'eht'], [1, 0])

    def test_text_alphabet_is_case_insensitive_by_default(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        self.assert_counts(['The Cat SAT'], ['the', 'cat', 'sat'], [1, 1, 1])

    def test_case_sensitive_alphabet(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ', case_sensitive=True)
        self.assert_counts(['The cat sat on the Mat'], ['the', 'The', 'at', 'Mat'], [1, 1, 3, 1])

    def test_characters_outside_the_alphabet_break_a_match(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyz')
        # The space is not in the alphabet, so it resets the search
        self.assert_counts(['ab cd'], ['abcd', 'ab', 'cd'], [0, 1, 1])

    def test_custom_complement_is_used_for_canonicalisation(self):
        set_alphabet(Alphabet('augc', complement='uacg'))
        # 'cau' is the reverse complement of 'aug', so both spellings count the same
        # segment: 'aug' at 0 and 4, plus 'cau' at 3
        self.assert_counts(['augcaugc'], ['aug'], [3])

    def test_segments_outside_the_alphabet_never_match(self):
        # A segment holding characters the alphabet does not have cannot match. It used
        # to be inserted with those characters stripped, and a segment stripped to
        # nothing matched at every position in the text.
        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        self.assert_counts(['the cat sat'], ['123', 'ca7', 'cat'], [0, 0, 1])

    def test_empty_segments_never_match(self):
        self.assert_counts(['aactg'], ['', 'aac'], [0, 1])

    def test_dna_segments_outside_the_alphabet_never_match(self):
        self.assert_counts(['aactgccagg'], ['xyz', 'aactg'], [0, 1])

    def test_read_error_placeholders_can_be_matched(self):
        # 'n' stands for a read error, and segments built with keep_read_error contain
        # it, so it has to be matchable rather than treated as a foreign character
        self.assert_counts(['aacngccagg'], ['acng', 'acgg'], [1, 0])

    def test_punctuation_can_be_part_of_the_alphabet(self):
        set_alphabet('abc=+<>')
        self.assert_counts(['a=b+c', 'a=b', 'c<>a=b'], ['a=b', '=b+', '<>', 'ccc'], [3, 1, 1, 0])

    def test_counting_survives_an_alphabet_change(self):
        self.assert_counts(['aactg'], ['aactg'], [1])
        set_alphabet('abcdefghijklmnopqrstuvwxyz')
        self.assert_counts(['aactg'], ['aactg'], [1])
        self.assert_counts(['aactg'], ['cagtt'], [0])


if __name__ == '__main__':
    unittest.main()
