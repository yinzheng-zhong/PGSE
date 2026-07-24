import numpy as np

from pgse.etc.alphabet import get_alphabet
from pgse.genome import canonicalize
from pgse.genome import get_complement
from pgse.genome.kmer import Kmer
from pgse.algos import aho_corasick

class Sequence:
    def __init__(
            self,
            filepath: str,
            keep_read_error: bool = False,
            concatenate_nodes: bool = False
    ) -> None:
        self.filepath: str = filepath
        self.keep_read_error: bool = keep_read_error
        self.concatenate_nodes: bool = concatenate_nodes
        self._km: Kmer = Kmer(keep_read_error=keep_read_error)
        self._nodes: list[str] = []
        self._complement_nodes: list[str] = []
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

    def _read_sequence(self) -> None:
        with open(self.filepath, 'r') as f:
            string = f.read().split('\n')

        # find the indices of all headers
        headers = [i for i, row in enumerate(string) if row.startswith('>')]

        # read the contigs between the headers
        contigs_multi_rows = [string[i+1:j] for i, j in zip(headers, headers[1:]+[None])]

        # concatenate the contigs, fold the case if the alphabet is case-insensitive,
        # and drop (or flag) anything outside the alphabet
        alphabet = get_alphabet()
        contigs = [
            alphabet.sanitise(''.join(contig), keep_read_error=self.keep_read_error)
            for contig in contigs_multi_rows
        ]

        if self.concatenate_nodes:
            self._nodes = [''.join(contigs)]
            self._complement_nodes = [get_complement(contigs[0])]
        else:
            self._nodes = contigs
            self._complement_nodes = [get_complement(contig) for contig in contigs]

    def get_kmer_count(self, k: int) -> np.ndarray:
        """
        Bin count for k-mers across all contigs. Faster than the lookup table with sequence matching.
        :param k: int: The length of the k-mers.
        :param no_consecutive: bool: Deprecated.
        """
        n = self._km.base ** k  # number of possible k-mers

        # Iterate through each node and count k-mers
        counts = [
            self._km.kmer_mapping(canonicalize(node[i:i + k]))
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
