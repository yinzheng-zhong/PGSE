import time
import os
from src.genome import km
from src.log import logger

class SegmentPool:
    def __init__(
            self,
    ):
        self.segments = []
        self.last_length = 0
        self.current_max_length = 0

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        return self.segments[index]

    def get_copy(self):
        """
        Return a copy of the segments table
        """
        return self.segments.copy()

    def add_all_kmer(self, k: int, keep_read_error=False):
        base = 5 if keep_read_error else 4
        kmers = [km.reverse_kmer_mapping(i, k) for i in range(base ** k)]
        self.add_subsequences(kmers, k, remove_duplicates=False)

    def use_subset(self, indices: [int]):
        """
        Filter the segments table by the keys
        """
        try:
            self.segments = [self.segments[i] for i in indices]
        except IndexError:
            logger.error('Index out of range')
            self.segments = []

        logger.info(f'Keeping {len(self.segments)} segments as shown below:\n{self.segments}')

    def save(self, filename: str):
        """
        Save the segments table to a file
        :param filename: str: The name of the file to save the lookup table
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            for item in self.segments:
                f.write("%s\n" % item)

        logger.info(f'Saved {len(self.segments)} segments to {filename}')

    def load(self, filename: str):
        """
        Load the segments table from a file
        :param filename: str: The name of the file to load the lookup table
        """
        with open(filename, 'r') as f:
            self.segments = f.read().split('\n')

        logger.info(f'Loaded {len(self.segments)} segments from {filename}')

        # figure out the max length
        self.current_max_length = max([len(s) for s in self.segments])
        logger.info(f'Set current max length: {self.current_max_length}')

    def add_subsequences(self, sequences: [str], current_length: int, remove_duplicates=True):
        """
        Add a list of sequences to the lookup table
        Note: This method uses set to remove duplicates, but it changes order of the sequences
        :param sequences: list: The list of sequences to add
        :param current_length: int: The current max length of the sequences
        :param remove_duplicates: bool: Remove duplicates from the list
        """
        self.segments = self.segments + sequences
        if remove_duplicates:
            self.segments = list(set(self.segments))

        self.last_length = self.current_max_length
        self.current_max_length = current_length

        logger.info(f'Number of segments: {len(self.segments)}')
        logger.info(f'Current max length: {self.current_max_length}')

    def redundant_elimination(self, importance_ranking: [int]):
        """
        If segments are substrings of other segments, keep the one with the highest importance
        :param importance_ranking: list: The list of indices of the most important features (descending order)
        :return:
        """

        # Result list to store non-substring strings
        result = []

        # Sort the segments based on the importance ranking
        ranked_segments = [self.segments[i] for i in importance_ranking]

        # Iterate through the ranked segments and check if any segment is a substring of another
        blocked = {}
        for i in range(len(ranked_segments)):
            if i in blocked:
                continue
            master_sub = {i}
            for j in range(len(ranked_segments)):
                if j in blocked:
                    continue
                if i != j and (ranked_segments[i] in ranked_segments[j] or ranked_segments[j] in ranked_segments[i]):
                    master_sub.add(j)

            # keep the one has the highest importance
            result.append(ranked_segments[min(master_sub)])
            # block the rest
            for k in master_sub - {min(master_sub)}:
                blocked[k] = True

        # Update the segments with the pruned list
        self.segments = result

        logger.info(f'Number of segments after redundant elimination: {len(self.segments)}')


if __name__ == '__main__':
    seg_manager = SegmentPool()
    seg_manager.segments = ['aaa', 'aa', 'aa', 'a', 'ab', 'b', 'ba', 'bac', 'c', 'ca', 'cab', 'd', 'da', 'dac', 'e',
                            'ea', 'eac']
    seg_manager.redundant_elimination([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    print()
