# Importing required packages for the test
import unittest

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
from pgse.model.util import oversample_minority_class


class TestUtil(unittest.TestCase):

    def setUp(self):
        self.ros = RandomOverSampler(random_state=42)

    def test_oversample_minority_class(self):
        # Test without shuffling
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 1, 1, 1, 0, 2])
        X_res, y_res = oversample_minority_class(X, y, shuffle_data=False)
        self.assertEqual(X_res.shape[0], 9)


if __name__ == '__main__':
    unittest.main()
