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
        self._nodes = []
        self._read_sequence()

        self.cache = Cache(len(self._nodes))

    def __len__(self):
        return sum(len(contig) for contig in self._nodes)

    def __getitem__(self, index):
        for contig in self._nodes:
            if index < len(contig):
                return contig[index]
            index -= len(contig)
        raise IndexError("Index out of range")

    def __str__(self):
        return ''.join(self._nodes)

    def len_nodes(self):
        return len(self._nodes)

    def _read_sequence(self):
        with open(self.filepath, 'r') as f:
            string = f.read().split('\n')

        # use filter() to remove header and empty lines
        contigs = list(filter(lambda x: not x.startswith('>') and x != '', string))

        # change all contigs to lower case
        contigs = [contig.lower() for contig in contigs]

        if self.keep_read_error:
            # change any character other than 'a', 't', 'g', 'c' to 'n' in each contig
            contigs = [''.join([c if c in 'atgc' else 'n' for c in contig]) for contig in contigs]
        else:
            # remove any character other than 'a', 't', 'g', 'c' in each contig
            contigs = [''.join([c for c in contig if c in 'atgc']) for contig in contigs]

        self._nodes = contigs

    def get_kmer_count(self, k: int, no_consecutive: bool):
        """
        Bin count for k-mers across all contigs. Faster than the lookup table with sequence matching.
        :param k: int: The length of the k-mers.
        :param no_consecutive: bool: Deprecated.
        """
        base = 5 if self.keep_read_error else 4
        n = base ** k  # number of possible k-mers

        # Iterate through each node and count k-mers
        counts = [
            km.kmer_mapping(km.canonical_reverse_complement(node[i:i + k]))
            for node in self._nodes if len(node) >= k
            for i in range(len(node) - k + 1)
        ]

        kmer_count = np.bincount(counts, minlength=n).astype(np.int32)

        return kmer_count

    def get_count_from_seg_manager(self, seg_pool_, no_consecutive=False):
        """
        Given a kmer sequence, return the transition frequency matrix. Cache is only available for the overlapping
        count for now.
        :param seg_pool_: SegmentPool: The SegmentPool instance.
        :param no_consecutive: bool: Removed
        """

        ext = seg_pool_.current_max_length - seg_pool_.last_length  # for pre-caching purposes

        counts = [
            self._occurrences_overlapping_cache(
                node, km.canonical_reverse_complement(seg), ext, node_id
            )
            for node_id, node in enumerate(self._nodes)
            for seg in seg_pool_
        ]
        seq_counts = np.array(counts).reshape(len(self._nodes), -1)
        self.cache.refresh()

        seq_count = np.sum(seq_counts, axis=0, dtype=np.int32)
        return seq_count

    def _occurrences_overlapping_cache(self, string, sub, ext, node_id):
        """
        Count the occurrences of a substring in a string. Use cache to store the indices of the substring.
        :param string: master string (contig in this case)
        :param sub: substring
        :param ext: the extension length
        :return:
        """
        cached = self.cache.get(sub, node_id)
        # cache hit
        if cached:
            starts = cached['indices']
            _ = [self._pre_cache(start, start + len(sub), ext, node_id) for start in starts]
            return cached['count']

        # cache miss
        count = start = 0
        while True:
            start = string.find(sub, start)
            if start >= 0:
                self.cache.set(sub, start, node_id)
                self._pre_cache(start, start + len(sub), ext, node_id)
                count += 1
                start += 1
            else:
                return count

    def _pre_cache(self, start, end, length, node_id):
        """
        Pre-cache the extensions of the substring.
        :param start:
        :param end:
        :param length:
        :return:
        """
        _ = [
            self.cache.set(self._nodes[node_id][j: j + end - (start - i)], j, node_id)
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
