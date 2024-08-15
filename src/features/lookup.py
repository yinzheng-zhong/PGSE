import time

from src.genome import km
from src.log import logger


class Lookup:
    """
    A class to represent a lookup table. Feature_ID: in64 -> Subsequence: str
    """

    def __init__(
            self,
    ):
        """
        Constructor for the Lookup class
        :param kmer: int: The k-mer value
        """
        self.lookup = []

    def __iter__(self):
        return iter(self.lookup)

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, index):
        return self.lookup[index]

    def get_copy(self):
        """
        Return a copy of the lookup table
        """
        return self.lookup.copy()

    def add_all_kmer(self, k: int, keep_read_error=False):
        base = 5 if keep_read_error else 4
        kmers = [km.reverse_kmer_mapping(i, k) for i in range(base ** k)]
        self.add_subsequences(kmers, remove_duplicates=False)

    def select(self, indices: [int]):
        """
        Filter the lookup table by the keys
        """
        try:
            self.lookup = [self.lookup[i] for i in indices]
        except IndexError:
            logger.error('Index out of range')
            self.lookup = []

        logger.info(f'Keeping {len(self.lookup)} segments as shown below:\n{self.lookup}')

    def add_subsequence(self, subsequence: str):
        """
        Add a subsequence to the lookup table
        :param subsequence: str: The subsequence to add
        """
        self.lookup = list(set(self.lookup + [subsequence]))
        logger.info(f'Number of segments: {len(self.lookup)}')

    def add_subsequences(self, sequences: [str], remove_duplicates=True):
        """
        Add a list of sequences to the lookup table
        Note: This method uses set to remove duplicates, but it changes order of the sequences
        :param sequences: list: The list of sequences to add#
        :param remove_duplicates: bool: Remove duplicates from the list
        """
        self.lookup = self.lookup + sequences
        if remove_duplicates:
            self.lookup = list(set(self.lookup))

        logger.info(f'Number of segments: {len(self.lookup)}')
