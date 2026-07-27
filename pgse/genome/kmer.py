from itertools import product
from typing import Optional

import numpy as np

from pgse.etc.alphabet import Alphabet, get_alphabet
from pgse.genome.utils import canonicalize


class Kmer:
    def __init__(
            self,
            keep_read_error: bool = False,
            alphabet: Optional[Alphabet] = None
    ) -> None:
        """
        :param keep_read_error: bool: Encode unexpected characters as the alphabet's
            unknown placeholder instead of dropping them.
        :param alphabet: Alphabet: Pin this instance to a specific alphabet. When
            omitted, the alphabet active at call time is used.
        """
        self.keep_read_error: bool = keep_read_error
        self._alphabet: Optional[Alphabet] = alphabet

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet if self._alphabet is not None else get_alphabet()

    @property
    def nuc_map(self) -> dict[str, int]:
        """Character to integer code map, e.g. {'a': 0, 't': 1, 'g': 2, 'c': 3}."""
        return self.alphabet.encoding_map(self.keep_read_error)

    @property
    def nucs(self) -> list[str]:
        """The characters segments are built from."""
        return self.alphabet.characters(self.keep_read_error)

    @property
    def base(self) -> int:
        """The base of the k-mer encoding, i.e. the number of distinct characters."""
        return self.alphabet.base(self.keep_read_error)

    def kmer_mapping(self, sequence: str) -> np.integer:
        """
        'aa' = 0 and 'at' = 1. 'aaa' also = 0
        :param sequence:
        :return:
        """
        k = len(sequence)  # Determine the length of the sequence
        nuc_map = self.nuc_map

        multiply_by = self.base ** np.arange(k - 1, -1, -1)  # Create the exponents for each position in the sequence
        value = np.dot([nuc_map[c] for c in sequence], multiply_by)  # Convert the sequence to an integer

        return value

    def gen_canonical_kmers(self, k: int) -> list[str]:
        """
            Generate a set of all canonical k-mers of length k using the provided canonicalize() function.

            Alphabets without a complement have no canonical form, so every k-mer is kept.

            :param k: int: The length of k-mers to generate.
            :return: set: A set of unique canonical k-mers.
        """
        kmers: list[str] = []

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
    ) -> str:
        nuc_map = self.nuc_map
        reverse_nuc_map = {v: k for k, v in nuc_map.items()}

        base = self.base

        sequence: list[str] = []

        while value > 0:
            index = value % base
            sequence.append(reverse_nuc_map[index])
            value = value // base

        # If the sequence is shorter than expected, pad with the first character
        # of the alphabet (0 value in the map)
        padding = reverse_nuc_map[0]
        while len(sequence) < k:
            sequence.append(padding)

        return ''.join(sequence[::-1])

    def random_sequence(self, length: int) -> str:
        """
        Generate a random sequence of a given length.
        :param length: int: The length of the sequence.
        :return: str: The random sequence.
        """
        sequence = np.random.choice(self.nucs, length)
        return ''.join(sequence)
