import numpy as np
import time

from src.genome import km, canonicalize
from src.genome.cache import Cache
from src.genome import get_complement
import ctypes


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

    def _load_lib(self):
        lib = None

        possible_locations = [
            '../../c/count_substrings.so',
            '../c/count_substrings.so',
            'c/count_substrings.so'
        ]
        # try to load the library
        for location in possible_locations:
            try:
                lib = ctypes.CDLL(location)
                break
            except OSError:
                pass

        lib.count_substrings.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p
        ]
        lib.count_substrings.restype = ctypes.c_int

        return lib

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

    def get_count_from_seg_manager(self, seg_pool_):
        """
        Given a kmer sequence, return the transition frequency matrix. Cache is only available for the overlapping
        count for now.
        :param seg_pool_: SegmentPool: The SegmentPool instance.
        """

        lib = self._load_lib()  # load the library here for Ray compatibility
        if lib is None:
            print('C library not found. Using Python implementation.')
            counting_method = self._occurrences_overlapping_py
        else:
            counting_method = self._occurrences_overlapping

        seq_counts = np.fromiter(
            (
                counting_method(
                    node, seg, lib
                )
                for node in self._nodes
                for seg in seg_pool_
            ),
            dtype=np.int32,
            count=len(self._nodes) * len(seg_pool_)
        ).reshape(len(self._nodes), len(seg_pool_))

        seq_count = np.sum(seq_counts, axis=0, dtype=np.int32)

        return seq_count

    def _occurrences_overlapping(self, genome, canonical_sub, lib):
        # return 0 if the canonical_sub is not truly canonical. Prevent errors from the seg_pool
        if get_complement(canonical_sub) < canonical_sub:
            return 0

        count = lib.count_substrings(genome.encode('utf-8'), canonical_sub.encode('utf-8'))

        # get the complement of the canonical substring
        complement_sub = get_complement(canonical_sub)

        # prevent palindrome sequences or errors.
        if complement_sub <= canonical_sub:
            return count

        count += lib.count_substrings(genome.encode('utf-8'), complement_sub.encode('utf-8'))

        return count

    def _occurrences_overlapping_py(self, genome, canonical_sub, lib):
        """
        Python implementation.
        :param genome: master string (contig in this case)
        :param canonical_sub: substring
        :return:
        """
        if get_complement(canonical_sub) < canonical_sub:
            return 0

        count = 0
        start = 0
        while True:
            start = genome.find(canonical_sub, start)
            if start >= 0:
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
                count += 1
                start += 1
            else:
                break

        return count