import itertools
from src.features import lookup
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

    def _extend_one_seq(self, sequence: str, length: int):
        # generate all possible extensions
        extensions = [''.join(p) for p in itertools.product(self.nucleotides, repeat=length)]

        extended_sequences = [extension[:i] + sequence + extension[i:]
                              for extension in extensions
                              for i in range(len(extension) + 1)]

        return extended_sequences

    def extend_all_seq_in_lookup(self, length: int):
        new = [self._extend_one_seq(sequence, length) for sequence in lookup]
        # reshape the list of lists to a single list
        new = [item for sublist in new for item in sublist]

        logger.info(f'Adding {len(new)} new segments to manager')

        lookup.add_subsequences(new)


if __name__ == '__main__':
    extender = Extender()
    o = extender._extend_one_seq('aaaaaaaa', 4)
    print(lookup)
    print(len(lookup))
    print(lookup[0])
