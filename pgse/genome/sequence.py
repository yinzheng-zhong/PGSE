from typing import Optional

import numpy as np

from pgse.dataset.alphabet import get_alphabet
from pgse.genome import canonicalize
from pgse.genome import get_complement
from pgse.genome.kmer import Kmer
from pgse.genome.utils import parse_fasta
from pgse.algos import aho_corasick

class Sequence:
    def __init__(
            self,
            filepath: Optional[str] = None,
            keep_read_error: bool = False,
            concatenate_nodes: bool = False,
            text: Optional[str] = None
    ) -> None:
        """
        Args:
            filepath: Path of the FASTA file to read. Ignored when text is given.
            keep_read_error: Replace characters outside the alphabet with the
                placeholder instead of dropping them.
            concatenate_nodes: Join every contig into a single node.
            text: Sequence data held in memory, read instead of a file.
        """
        if filepath is None and text is None:
            raise ValueError('A Sequence needs either a filepath or text.')

        self.filepath: Optional[str] = filepath
        self.text: Optional[str] = text
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

    @property
    def nodes(self) -> list[str]:
        """The contigs/scaffolds, already case-normalised and restricted to the alphabet."""
        return self._nodes

    def _read_sequence(self) -> None:
        if self.text is not None:
            text = self.text
        else:
            with open(str(self.filepath), 'r') as f:
                text = f.read()

        # fold the case if the alphabet is case-insensitive, and drop (or flag)
        # anything outside the alphabet
        alphabet = get_alphabet()
        contigs = [
            alphabet.sanitise(contig, keep_read_error=self.keep_read_error)
            for contig in parse_fasta(text)
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
