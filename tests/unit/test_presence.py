import unittest

import numpy as np
import scipy.sparse as sp

from pgse.dataset.counts import to_presence

COUNTS = [[0, 3, 1], [7, 0, 0]]
PRESENCE = [[0, 1, 1], [1, 0, 0]]


class TestToPresence(unittest.TestCase):
    """Counts become 0/1 presence, in place, keeping the matrix's type and dtype."""

    def test_dense_float32(self) -> None:
        matrix = np.array(COUNTS, dtype=np.float32)

        result = to_presence(matrix)

        self.assertIs(result, matrix)
        self.assertEqual(np.float32, result.dtype)
        np.testing.assert_array_equal(PRESENCE, result)

    def test_dense_uint16(self) -> None:
        matrix = np.array(COUNTS, dtype=np.uint16)

        result = to_presence(matrix)

        self.assertEqual(np.uint16, result.dtype)
        np.testing.assert_array_equal(PRESENCE, result)

    def test_sparse_keeps_zeros_unstored(self) -> None:
        matrix = sp.csr_matrix(np.array(COUNTS, dtype=np.float32))
        stored = matrix.nnz

        result = to_presence(matrix)

        self.assertIs(result, matrix)
        self.assertTrue(sp.issparse(result))
        self.assertEqual(stored, result.nnz)
        np.testing.assert_array_equal(PRESENCE, result.toarray())


if __name__ == '__main__':
    unittest.main()
