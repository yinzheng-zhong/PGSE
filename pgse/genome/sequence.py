import numpy as np

from pgse.genome import km, canonicalize
from pgse.genome import get_complement
from pgse.algos import aho_corasick

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

    def get_kmer_count(self, k: int):
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

    def get_count_from_seg_manager(self, seg_pool_):
        """
        Given a kmer sequence, return the transition frequency matrix.
        :param seg_pool_: SegmentPool: The SegmentPool instance.
        """
        seg_count = aho_corasick.count_segments(self._nodes, seg_pool_)

        return seg_count
