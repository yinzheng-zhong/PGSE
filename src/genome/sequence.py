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
            '../../c/count_segments.so',
            '../c/count_segments.so',
            'c/count_segments.so',
        ]
        # try to load the library
        for location in possible_locations:
            try:
                lib = ctypes.CDLL(location)
                break
            except OSError:
                pass

        if lib is not None:
            lib.count_segments.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),  # nodes
                ctypes.c_int,  # num_nodes
                ctypes.POINTER(ctypes.c_char_p),  # segments
                ctypes.c_int,  # num_segments
                ctypes.POINTER(ctypes.c_int)  # result_counts
            ]
            lib.count_segments.restype = None

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
        lib = self._load_lib()
        if lib is None:
            print('C library not found. Using Python implementation.')
            counting_method = self._occurrences_overlapping_py
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

            seq_count = np.sum(seq_counts, axis=0)
        else:
            # Prepare data for C function
            num_nodes = len(self._nodes)
            num_segments = len(seg_pool_)

            # Create arrays of c_char_p
            node_array = (ctypes.c_char_p * num_nodes)(*(node.encode('utf-8') for node in self._nodes))
            segment_array = (ctypes.c_char_p * num_segments)(*(seg.encode('utf-8') for seg in seg_pool_))

            # Prepare result array
            result_counts = (ctypes.c_int * num_segments)()

            # Call the C function
            lib.count_segments(
                node_array, ctypes.c_int(num_nodes),
                segment_array, ctypes.c_int(num_segments),
                result_counts
            )

            # Convert result to NumPy array
            seq_count = np.ctypeslib.as_array(result_counts, shape=(num_segments,))

        return seq_count

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