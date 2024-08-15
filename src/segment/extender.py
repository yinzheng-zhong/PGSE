import itertools

from src.genome import seq_manager
from src.segment import seg_pool
from src.log import logger


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
        # should only be made on the segments that have the maximum length as earlier segments are already extended
        new = [self._extend_one_seg(sequence, length) for sequence in seg_pool if len(sequence) == seg_pool.current_max_length]
        # reshape the list of lists to a single list
        new = [item for sublist in new for item in sublist]

        logger.info(f'Adding {len(new)} new subsequences to lookup')

        seg_pool.add_subsequences(new, current_length=seg_pool.current_max_length + length)


if __name__ == '__main__':
    extender = Extender()
    o = extender._extend_one_seg('aaaaaaaa', 4)
    print(seg_pool)
    print(len(seg_pool))
    print(seg_pool[0])
