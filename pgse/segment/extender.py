import itertools

from pgse.genome import canonicalize
from pgse.segment import seg_pool
from pgse.log import logger
from pgse.segment.util import remove_duplicate_elements


class Extender:
    def __init__(
            self,
            keep_read_error: bool = False
    ):
        self.keep_read_error = keep_read_error
        self.nucleotides = ['a', 't', 'g', 'c']
        if self.keep_read_error:
            self.nucleotides.append('n')

    def _extend_one_seg(self, sequence: str, length: int):
        # generate all possible extensions
        extensions = [''.join(p) for p in itertools.product(self.nucleotides, repeat=length)]

        extended_sequences = [extension[:i] + sequence + extension[i:]
                              for extension in extensions
                              for i in range(len(extension) + 1)]

        return extended_sequences

    def extend_all_segs(self, length: int):
        # extended if their lengths fall between the last length and the sum of last length and the specified length.
        # Make it suitable for grafting.
        new = [
            self._extend_one_seg(sequence, i) for sequence in seg_pool for i in range(1, length + 1) if
            len(sequence) in range(seg_pool.last_length + 1, seg_pool.last_length + length + 1)
        ]

        # reshape the list of lists to a single list
        new = [item for sublist in new for item in sublist]

        canonical_sequences = [canonicalize(seq) for seq in new]
        # remove duplicates
        canonical_sequences = remove_duplicate_elements(canonical_sequences)

        if len(canonical_sequences) > 0:
            logger.info(f'Adding {len(canonical_sequences)} new canonical segments to the pool')
        else:
            logger.warning('No new segments to add. Finished extending all segments')
            raise ValueError('No new segments to add')

        seg_pool.add_subsequences(canonical_sequences, current_length=seg_pool.current_max_length + length)
        logger.info(f'Current max length: {seg_pool.current_max_length}')
        logger.warning("The order of the segments might have changed")


if __name__ == '__main__':
    seg_pool.segments = ['m']
    extender = Extender()
    o = extender.extend_all_segs(4)
    print(seg_pool)
    print(len(seg_pool))
    print(seg_pool[0])
