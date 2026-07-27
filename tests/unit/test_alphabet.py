import unittest

from pgse.etc.alphabet import Alphabet, DNA, get_alphabet, reset_alphabet, set_alphabet


class TestAlphabet(unittest.TestCase):
    def tearDown(self):
        reset_alphabet()

    def test_default_is_dna(self):
        self.assertEqual(('a', 't', 'g', 'c'), DNA.chars)
        self.assertFalse(DNA.case_sensitive)
        self.assertTrue(DNA.has_complement)
        self.assertEqual('n', DNA.unknown_char)
        self.assertEqual(DNA, get_alphabet())

    def test_dna_reverse_complement(self):
        self.assertEqual('gcat', DNA.get_complement('atgc'))
        self.assertEqual('atgc', DNA.canonicalize('gcat'))
        self.assertEqual('aaaa', DNA.canonicalize('tttt'))

    def test_complement_leaves_the_unknown_placeholder_alone(self):
        self.assertEqual('cnt', DNA.get_complement('ang'))

    def test_alphabet_without_complement_is_not_canonicalised(self):
        text = Alphabet('abcdefghijklmnopqrstuvwxyz ')
        self.assertFalse(text.has_complement)
        # Returning the sequence unchanged makes canonicalisation a no-op
        self.assertEqual('hello', text.get_complement('hello'))
        self.assertEqual('hello', text.canonicalize('hello'))

    def test_custom_complement(self):
        rna = Alphabet('augc', complement='uacg')
        self.assertEqual('cau', rna.get_complement('aug'))
        self.assertEqual('aug', rna.canonicalize('cau'))

    def test_complement_pairs_only_have_to_be_given_once(self):
        pairs = Alphabet('abcd', complement={'a': 'b', 'c': 'd'})
        self.assertEqual({'a': 'b', 'b': 'a', 'c': 'd', 'd': 'c', 'n': 'n'}, pairs.complement_map)
        # 'abc' -> complement 'bad' -> reversed 'dab'
        self.assertEqual('dab', pairs.get_complement('abc'))

    def test_case_insensitive_by_default(self):
        self.assertEqual('atgc', DNA.normalise('ATGC'))
        self.assertEqual('atgc', DNA.sanitise('ATGC'))

    def test_case_sensitive_keeps_both_cases(self):
        alphabet = Alphabet('abAB', case_sensitive=True)
        self.assertEqual(('a', 'b', 'A', 'B'), alphabet.chars)
        self.assertEqual('AaBb', alphabet.normalise('AaBb'))
        self.assertEqual('AaBb', alphabet.sanitise('AaBbXx'))

    def test_case_insensitive_folds_the_alphabet_itself(self):
        alphabet = Alphabet('aAbB')
        self.assertEqual(('a', 'b'), alphabet.chars)
        self.assertEqual('ab', alphabet.sanitise('AB'))

    def test_sanitise_drops_unknown_characters(self):
        text = Alphabet('abcdefghijklmnopqrstuvwxyz')
        self.assertEqual('thecatsat', text.sanitise('The cat, 42 sat!'))

    def test_sanitise_can_keep_read_errors(self):
        self.assertEqual('atnnnnngc', DNA.sanitise('atxxxxxgc', keep_read_error=True))

    def test_keeping_read_errors_needs_a_placeholder(self):
        alphabet = Alphabet('abcn')
        self.assertIsNone(alphabet.unknown_char)
        with self.assertRaises(ValueError):
            alphabet.sanitise('abcx', keep_read_error=True)

    def test_explicit_placeholder(self):
        alphabet = Alphabet('abcn', unknown_char='?')
        self.assertEqual('abc?', alphabet.sanitise('abcx', keep_read_error=True))

    def test_encoding_map_follows_the_given_order(self):
        self.assertEqual({'a': 0, 't': 1, 'g': 2, 'c': 3}, DNA.encoding_map())
        self.assertEqual({'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4}, DNA.encoding_map(include_unknown=True))
        self.assertEqual(4, DNA.base())
        self.assertEqual(5, DNA.base(include_unknown=True))

    def test_rejects_an_empty_alphabet(self):
        with self.assertRaises(ValueError):
            Alphabet('')

    def test_rejects_a_complement_of_the_wrong_length(self):
        with self.assertRaises(ValueError):
            Alphabet('atgc', complement='ta')

    def test_rejects_a_complement_outside_the_alphabet(self):
        with self.assertRaises(ValueError):
            Alphabet('atgc', complement='tacx')

    def test_rejects_an_irreversible_complement(self):
        with self.assertRaises(ValueError):
            Alphabet('atgc', complement='tgca')

    def test_rejects_a_placeholder_that_clashes_with_the_alphabet(self):
        with self.assertRaises(ValueError):
            Alphabet('atgcn', unknown_char='n')

    def test_set_alphabet_accepts_a_string(self):
        alphabet = set_alphabet('xyz')
        self.assertEqual(('x', 'y', 'z'), alphabet.chars)
        self.assertEqual(alphabet, get_alphabet())

    def test_set_alphabet_accepts_an_instance(self):
        alphabet = Alphabet('xyz', case_sensitive=True)
        self.assertEqual(alphabet, set_alphabet(alphabet))
        self.assertEqual(alphabet, get_alphabet())

    def test_reset_alphabet(self):
        set_alphabet('xyz')
        self.assertEqual(DNA, reset_alphabet())

    def test_byte_tables_for_the_c_library(self):
        index, size, complement = DNA.byte_tables()
        self.assertEqual(5, size)  # the four bases plus the unknown placeholder
        self.assertEqual(0, index[ord('a')])
        self.assertEqual(0, index[ord('A')])  # case-insensitive alphabets accept both
        self.assertEqual(4, index[ord('n')])
        self.assertEqual(-1, index[ord('x')])
        self.assertEqual(ord('t'), complement[ord('a')])
        self.assertEqual(ord('n'), complement[ord('n')])

    def test_byte_tables_without_a_complement(self):
        _, size, complement = Alphabet('abc', unknown_char=None).byte_tables()
        self.assertEqual(3, size)
        self.assertIsNone(complement)

    def test_case_sensitive_byte_tables_separate_the_cases(self):
        index, size, _ = Alphabet('abAB', case_sensitive=True, unknown_char=None).byte_tables()
        self.assertEqual(4, size)
        self.assertNotEqual(index[ord('a')], index[ord('A')])

    def test_multi_byte_characters_are_not_byte_safe(self):
        self.assertTrue(DNA.is_byte_safe)
        self.assertFalse(Alphabet('αβγ').is_byte_safe)


if __name__ == '__main__':
    unittest.main()
