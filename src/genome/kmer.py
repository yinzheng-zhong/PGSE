import numpy as np


class Kmer:
    def __init__(
            self,
            keep_read_error=False
    ):
        self.keep_read_error = keep_read_error
        self.nuc_map = {'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4} if keep_read_error else {'a': 0, 't': 1, 'g': 2, 'c': 3}
        self.base = 5 if self.keep_read_error else 4

    def kmer_mapping(self, sequence):
        """
        'aa' = 0 and 'at' = 1. 'aaa' also = 0
        :param sequence:
        :return:
        """
        k = len(sequence)  # Determine the length of the sequence

        multiply_by = self.base ** np.arange(k - 1, -1, -1)  # Create the exponents for each position in the sequence
        value = np.dot([self.nuc_map[c] for c in sequence], multiply_by)  # Convert the sequence to an integer

        return value

    def reverse_kmer_mapping(
            self,
            value: int,
            k: int
    ):
        nuc_map = {'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4} if self.keep_read_error else {'a': 0, 't': 1, 'g': 2, 'c': 3}
        reverse_nuc_map = {v: k for k, v in nuc_map.items()}

        base = 5 if self.keep_read_error else 4

        sequence = []

        while value > 0:
            index = value % base
            sequence.append(reverse_nuc_map[index])
            value = value // base

        # If the sequence is shorter than expected, pad with 'a' (0 value in the map)
        while len(sequence) < k:
            sequence.append('a')

        return ''.join(sequence[::-1])
