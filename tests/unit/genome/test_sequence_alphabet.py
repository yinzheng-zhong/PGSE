import os
import unittest

from pgse.etc.alphabet import Alphabet, reset_alphabet, set_alphabet
from pgse.genome.sequence import Sequence
from pgse.segment.extender import Extender
from pgse.segment.segment_pool import SegmentPool

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'resource')
DNA_FILE = os.path.join(RESOURCES, 'test_sequence.txt')
TEXT_FILE = os.path.join(RESOURCES, 'test_text.txt')


class TestSequenceWithAlphabets(unittest.TestCase):
    def tearDown(self):
        reset_alphabet()

    def test_dna_is_read_as_before(self):
        sequence = Sequence(DNA_FILE)
        self.assertEqual('aactgccaggcatcaaattagat', str(sequence))

    def test_dna_drops_characters_outside_the_alphabet(self):
        # The FASTA headers and the line breaks never make it into the contigs
        self.assertNotIn('>', str(Sequence(DNA_FILE)))

    def test_text_alphabet_keeps_letters_and_spaces(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        sequence = Sequence(TEXT_FILE)
        self.assertEqual(2, sequence.len_nodes())
        self.assertEqual('the cat sat on the matit sat there quietly for  minutes', sequence._nodes[0])
        self.assertEqual('the dog barked', sequence._nodes[1])

    def test_case_sensitive_text_alphabet(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ', case_sensitive=True)
        sequence = Sequence(TEXT_FILE)
        self.assertEqual('The cat sat on the matIt sat there quietly for  minutes', sequence._nodes[0])

    def test_read_errors_use_the_unknown_placeholder(self):
        set_alphabet(Alphabet('abcdefghijklmnopqrstuvwxyz ', unknown_char='?'))
        sequence = Sequence(TEXT_FILE, keep_read_error=True)
        self.assertIn('??', sequence._nodes[0])  # the digits in "42"
        self.assertTrue(sequence._nodes[0].endswith('minutes?'))  # the exclamation mark

    def test_kmer_counts_use_the_alphabet_size(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        sequence = Sequence(TEXT_FILE)
        counts = sequence.get_kmer_count(1)
        self.assertEqual(27, len(counts))  # 26 letters plus the space
        self.assertEqual(len(str(sequence)), int(counts.sum()))

    def test_segment_counting_over_text(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        sequence = Sequence(TEXT_FILE)
        pool = SegmentPool()
        pool.segments = ['the', 'sat', 'dog', 'cat', 'zebra']
        # 'the' also matches inside 'there', and matches are counted across both nodes
        self.assertEqual([4, 2, 1, 1, 0], list(sequence.get_count_from_seg_manager(pool)))

    def test_extender_uses_the_alphabet(self):
        set_alphabet('abc')
        pool = SegmentPool()
        pool.segments = ['a']
        pool.last_length = 0
        pool.current_max_length = 1

        import pgse.segment.extender as extender_module
        original_pool = extender_module.seg_pool
        extender_module.seg_pool = pool
        try:
            Extender().extend_all_segs(1)
        finally:
            extender_module.seg_pool = original_pool

        # 'a' extended by one character on either side, over the three-letter alphabet
        self.assertEqual({'aa', 'ab', 'ac', 'ba', 'ca'}, set(pool.segments) - {'a'})


if __name__ == '__main__':
    unittest.main()
