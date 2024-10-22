import numpy as np
import time

from src.genome import km, canonicalize
from src.genome.cache import Cache
from src.genome import get_complement


class Sequence:
    def __init__(
            self,
            filepath: str,
            keep_read_error=False,
            concatenate_nodes=False
    ):
        self.filepath = filepath
        self.keep_read_error = keep_read_error
        self.concatenate_nodes = concatenate_nodes
        self._nodes = []
        self._complement_nodes = []
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

        # find the indices of all headers
        headers = [i for i, row in enumerate(string) if row.startswith('>')]

        # read the contigs between the headers
        contigs_multi_rows = [string[i+1:j] for i, j in zip(headers, headers[1:]+[None])]

        # concatenate the contigs and convert to lowercase
        contigs = [''.join(contig).lower() for contig in contigs_multi_rows]

        if self.keep_read_error:
            # change any character other than 'a', 't', 'g', 'c' to 'n' in each contig
            contigs = [''.join([c if c in 'atgc' else 'n' for c in contig]) for contig in contigs]
        else:
            # remove any character other than 'a', 't', 'g', 'c' in each contig
            contigs = [''.join([c for c in contig if c in 'atgc']) for contig in contigs]


        if self.concatenate_nodes:
            self._nodes = [''.join(contigs)]
            self._complement_nodes = [get_complement(contigs[0])]
        else:
            self._nodes = contigs
            self._complement_nodes = [get_complement(contig) for contig in contigs]

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
            km.kmer_mapping(canonicalize(node[i:i + k]))
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

        seq_counts = np.fromiter(
            (
                self._occurrences_overlapping_cache(
                    node, seg, ext, node_id
                )
                for node_id, node in enumerate(self._nodes)
                for seg in seg_pool_
            ),
            dtype=np.int32,
            count=len(self._nodes) * len(seg_pool_)
        ).reshape(len(self._nodes), len(seg_pool_))

        self.cache.refresh()

        seq_count = np.sum(seq_counts, axis=0, dtype=np.int32)

        return seq_count

    def _occurrences_overlapping_cache(self, genome, canonical_sub, ext, node_id):
        """
        Count the occurrences of a substring in a string. Use cache to store the indices of the substring.
        :param genome: master string (contig in this case)
        :param canonical_sub: substring
        :param ext: the extension length
        :return:
        """
        # return 0 if the canonical_sub is not truly canonical. Prevent errors from the seg_pool
        if get_complement(canonical_sub) < canonical_sub:
            return 0

        cached = self.cache.get(canonical_sub, node_id)
        # cache hit
        if cached:
            starts = cached['indices']
            _ = [self._pre_cache(start, start + len(canonical_sub), ext, node_id) for start in starts]
            return cached['count']

        # cache miss
        count = start = 0
        while True:
            start = genome.find(canonical_sub, start)
            if start >= 0:
                self.cache.set(canonical_sub, start, node_id)
                self._pre_cache(start, start + len(canonical_sub), ext, node_id)
                count += 1
                start += 1
            else:
                break

        # get the complement of the canonical substring
        complement_sub = get_complement(canonical_sub)

        # prevent palindrome sequences or errors.
        if complement_sub <= canonical_sub:
            return count

        start = 0
        while True:
            start = genome.find(complement_sub, start)
            if start >= 0:
                self.cache.set(complement_sub, start, node_id)
                self._pre_cache(start, start + len(complement_sub), ext, node_id)
                count += 1
                start += 1
            else:
                break

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

    def _occurrences_overlapping(self, genome, canonical_sub):
        # return 0 if the canonical_sub is not truly canonical
        if get_complement(canonical_sub) < canonical_sub:
            return 0

        count = start = 0

        # count the canonical substring in the genome first
        while True:
            start = genome.find(canonical_sub, start) + 1
            if start > 0:
                count += 1
            else:
                break

        # get the complement of the canonical substring
        complement_sub = get_complement(canonical_sub)

        # prevent palindrome sequences or errors.
        if complement_sub >= canonical_sub:
            return count

        # count the complement substring in the genome
        start = 0
        while True:
            start = genome.find(complement_sub, start) + 1
            if start > 0:
                count += 1
            else:
                break

        return count
