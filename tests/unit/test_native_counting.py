import unittest

import numpy as np
import scipy.sparse as sp

from pgse.algos import native_counter
from pgse.algos.aho_corasick_py import AhoCorasickPy
from pgse.dataset.alphabet import Alphabet, reset_alphabet, set_alphabet


class _FakeSeq:
    """Stand-in for Sequence: exposes a ``nodes`` list, which is all the counter needs."""

    def __init__(self, nodes):
        self._nodes = nodes

    @property
    def nodes(self):
        return self._nodes


@unittest.skipUnless(native_counter.native_available(), 'native counting kernel not built')
class TestNativeCounting(unittest.TestCase):
    """
    The Rust kernel must produce exactly the same counts as the pure-Python
    reference, for every alphabet feature (complement/canonicalisation, case folding,
    custom complements, out-of-alphabet handling) and for both storage layouts.
    """

    def tearDown(self):
        reset_alphabet()

    def _reference(self, segments, samples):
        py = AhoCorasickPy()
        return np.asarray(
            [py.count_segments(nodes, segments) for nodes in samples], dtype=np.int64
        )

    def _native(self, segments, samples, dtype=np.float32, sparse=True):
        matcher = native_counter.build_matcher(segments)
        seqs = [_FakeSeq(nodes) for nodes in samples]
        out = native_counter.count_matrix(
            matcher, seqs, len(segments),
            dtype=dtype, sparse=sparse, threads=2, desc='test', chunk_size=8,
        )
        return out.toarray() if sp.issparse(out) else out

    def assert_matches(self, segments, samples, dtype=np.float32, sparse=True):
        ref = self._reference(segments, samples)
        got = self._native(segments, samples, dtype=dtype, sparse=sparse)
        self.assertEqual(got.shape, ref.shape)
        np.testing.assert_array_equal(got.astype(np.int64), ref)

    def test_dna_counts_both_strands(self):
        set_alphabet('atgc')
        self.assert_matches(['aactg'], [['aactgccaggcagtt']])

    def test_dna_skips_non_canonical_segments(self):
        set_alphabet('atgc')
        self.assert_matches(['cagtt', 'aactg'], [['aactgccagg']])

    def test_text_case_insensitive(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        self.assert_matches(['the', 'cat', 'at', 'zebra'], [['The Cat SAT on the mat']])

    def test_case_sensitive(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ', case_sensitive=True)
        self.assert_matches(['the', 'The', 'at', 'Mat'], [['The cat sat on the Mat']])

    def test_custom_complement(self):
        set_alphabet(Alphabet('augc', complement='uacg'))
        self.assert_matches(['aug'], [['augcaugc']])

    def test_out_of_alphabet_and_empty_segments(self):
        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        self.assert_matches(['123', 'ca7', 'cat', ''], [['the cat sat']])

    def test_multiple_nodes_and_samples(self):
        set_alphabet('atgc')
        segments = ['at', 'gc', 'atgc', 'aa', 'cg']
        samples = [
            ['atatgc', 'gcgc'],
            ['aacgt'],
            [''],
            ['atgcatgcatgc'],
        ]
        self.assert_matches(segments, samples, dtype=np.float32, sparse=True)
        self.assert_matches(segments, samples, dtype=np.uint16, sparse=False)

    def test_uint16_saturation(self):
        set_alphabet(Alphabet('a', complement=None))
        got = self._native(['a'], [['a' * 70000]], dtype=np.uint16, sparse=True)
        self.assertEqual(int(got[0, 0]), np.iinfo(np.uint16).max)

    def test_dense_and_sparse_agree(self):
        set_alphabet('atgc')
        segments = ['at', 'gc', 'ta', 'cg', 'atg']
        samples = [['atgcatgc'], ['gcta'], ['aaa']]
        dense = self._native(segments, samples, sparse=False)
        sparse = self._native(segments, samples, sparse=True)
        np.testing.assert_array_equal(dense, sparse)


if __name__ == '__main__':
    unittest.main()
