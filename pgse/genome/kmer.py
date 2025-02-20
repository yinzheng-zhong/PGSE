from itertools import product

import numpy as np

from pgse.genome.utils import get_complement, canonicalize


class Kmer:
    def __init__(
            self,
            keep_read_error=False
    ):
        self.keep_read_error = keep_read_error
        self.nuc_map = {'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4} if keep_read_error else {'a': 0, 't': 1, 'g': 2, 'c': 3}
        self.nucs = list(self.nuc_map.keys())
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

    def gen_canonical_kmers(self, k):
        """
            Generate a set of all canonical k-mers of length k using the provided canonicalize() function.

            :param k: int: The length of k-mers to generate.
            :return: set: A set of unique canonical k-mers.
        """
        kmers = []

        # Iterate over all possible k-length tuples from self.nucs
        for kmer_tuple in product(self.nucs, repeat=k):
            # Convert the tuple to a string
            kmer = ''.join(kmer_tuple)

            # Canonicalize the k-mer
            can_kmer = canonicalize(kmer)

            # Add it to our set (duplicate canonical forms are automatically ignored)
            kmers.append(can_kmer)

        return list(dict.fromkeys(kmers))

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

    def random_sequence(self, length: int):
        """
        Generate a random sequence of a given length.
        :param length: int: The length of the sequence.
        :return: str: The random sequence.
        """
        sequence = np.random.choice(list(self.nuc_map.keys()), length)
        return ''.join(sequence)
