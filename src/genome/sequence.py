import numpy as np
import time

from src.genome import km
from src.genome.cache import Cache


class Sequence:
    def __init__(
            self,
            filepath: str,
            keep_read_error=False,
    ):
        self.filepath = filepath
        self.keep_read_error = keep_read_error
        self._sequence = ''
        self._read_sequence()

        self.cache = Cache()


    def __len__(self):
        return len(self._sequence)

    def __getitem__(self, index):
        return self._sequence[index]

    def __str__(self):
        return self._sequence

    def _read_sequence(self):
        with open(self.filepath, 'r') as f:
            string = f.read().split('\n')

        # use filter() to remove header and empty lines
        string = list(filter(lambda x: not x.startswith('>') and x != '', string))

        # join the list of strings into one string
        string = ''.join(string)

        # change to lower case
        string = string.lower()

        if self.keep_read_error:
            # change any character other than 'a', 't', 'g', 'c' to 'n'
            string = ''.join([c if c in 'atgc' else 'n' for c in string])
        else:
            # remove any character other than 'a', 't', 'g', 'c'
            string = ''.join([c for c in string if c in 'atgc'])

        self._sequence = string

    def get_kmer_count(self, k: int, no_consecutive: bool):
        """
        Bin count for k-mers. Faster than the lookup table with sequence matching. Used initially before any subsequence
        is selected.
        :param k: int: The length of the k-mers.
        :param no_consecutive: bool: Remove consecutive identical k-mers if True.
        """
        base = 5 if self.keep_read_error else 4
        n = base ** k  # number of possible k-mers

        # Directly map the sequence to integer values without a separate function
        kmer_seq = list(map(lambda i: km.kmer_mapping(self[i:i + k]), range(len(self) - k + 1)))

        kmer_seq = np.array(kmer_seq, dtype=np.int32)
        kmer_count = np.bincount(kmer_seq, minlength=n)

        return kmer_count

    def get_count_from_seg_manager(self, seg_pool_, no_consecutive):

        """
        Given a kmer sequence, return the transition frequency matrix. Cache is only available for the overlapping
        count for now.
        :param seg_pool_: SegmentPool: The SegmentPool instance.
        :param no_consecutive: bool: Remove consecutive identical k-mers if True.
        """
        if no_consecutive:
            seq_count = np.array([self._occurrences(self._sequence, seg) for seg in seg_pool_], dtype=np.int32)
        else:
            ext = seg_pool_.current_max_length - seg_pool_.last_length
            seq_count = np.array([self._occurrences_overlapping_cache(self._sequence, seg, ext) for seg in seg_pool_], dtype=np.int32)

            self.cache.refresh()

        return seq_count

    def _occurrences_overlapping_cache(self, string, sub, ext):
        """
        Count the occurrences of a substring in a string. Use cache to store the indices of the substring.
        :param string: master string
        :param sub: substring
        :param ext: the extension length
        :return:
        """
        cached = self.cache.get(sub)
        # cache hit
        if cached:
            starts = cached['indices']
            _ = [self._pre_cache(start, start + len(sub), ext) for start in starts]
            return cached['count']

        # cache miss
        count = start = 0
        while True:
            start = string.find(sub, start)
            if start >= 0:
                self.cache.set(sub, start)
                self._pre_cache(start, start + len(sub), ext)
                count += 1
                start += 1
            else:
                return count

    def _pre_cache(self, start, end, length):
        """
        Pre-cache the extensions of the substring.
        :param start:
        :param end:
        :param length:
        :return:
        """
        _ = [
            self.cache.set(self._sequence[j: j + end - (start - i)], j)
            for i in range(1, length + 1)
            for j in range(start - i, start + 1)
        ]

    def _occurrences_overlapping(self, string, sub):
        count = start = 0
        while True:
            start = string.find(sub, start) + 1
            if start > 0:
                count += 1
            else:
                return count

    def _occurrences(self, string, sub):
        return string.count(sub)
