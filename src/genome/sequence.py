import numpy as np
import time

from src.genome import km


class Sequence:
    def __init__(
            self,
            filepath: str,
            keep_read_error=False,
    ):
        self.filepath = filepath
        self.keep_read_error = keep_read_error
        self._sequence = self._read_sequence()

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

        return string

    def get_kmer_count(self, k: int):
        """
        Bin count for k-mers. Faster than the lookup table with sequence matching. Used initially before any subsequence
        is selected.
        """
        base = 5 if self.keep_read_error else 4
        n = base ** k  # number of possible k-mers

        # Directly map the sequence to integer values without a separate function
        kmer_seq = list(map(lambda i: km.kmer_mapping(self[i:i + k]), range(len(self) - k + 1)))
        kmer_seq = np.array(kmer_seq, dtype=np.int32)

        kmer_count = np.bincount(kmer_seq, minlength=n)

        return kmer_count

    def get_count_from_seg_manager(self, seg_pool_):

        """
        Given a kmer sequence, return the transition frequency matrix.
        """
        seq_count = np.array([Sequence._occurrences(self._sequence, seg) for seg in seg_pool_], dtype=np.int32)
        return seq_count

    @staticmethod
    def _occurrences(string, sub):
        count = start = 0
        while True:
            start = string.find(sub, start) + 1
            if start > 0:
                count += 1
            else:
                return count


if __name__ == '__main__':
    from src.segment import seg_pool
    seq = Sequence('../../../volatile/e_coli_mic/562.5419.fna')
    seg_pool.segments = ['aaaaaaaa'] * 2000
    start = time.time()
    o = seq.get_count_from_seg_manager(seg_pool)
    print('Time:', time.time() - start)
    print(o)