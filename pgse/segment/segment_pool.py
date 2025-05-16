import time
import os
import pickle
from pgse.genome import km
from pgse.log import logger
from pgse.segment.util import remove_duplicate_elements
import pandas as pd

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
        Return a copy of the segments table.
        :return: list: A copy of the segments list.
        """
        return self.segments.copy()

    def clear(self):
        """
        Clear the segments list.
        """
        self.segments = []
        self.last_length = 0
        self.current_max_length = 0

    def add_all_kmer(self, k: int, extension: int, keep_read_error=False):
        """
        Add all k-mers to the segments list.
        Note: although the counting of kmers is not based on strings added here, we still need to add them the list
        to keep track of the segments.
        :param extension:
        :param k: int: The length of the k-mers.
        :param keep_read_error: bool: Include read errors if True.
        """
        logger.info(f'Adding all {k}-mers to the segment pool')
        kmers = km.gen_canonical_kmers(k)

        self.current_max_length = k - extension
        self.add_subsequences(kmers, k, remove_duplicates=False)

    def use_subset(self, indices: [int]):
        """
        Filter the segments list by indices.
        :param indices: list of int: The indices to keep in the segments list.
        """
        try:
            self.segments = [self.segments[i] for i in indices]
        except IndexError:
            logger.error('Index out of range')
            self.segments = []

        logger.info(f'Keeping {len(self.segments)} segments as shown below:\n{self.segments[:100]}...')

    def save(self, filename: str):
        """
        Save the SegmentPool instance to a file using pickle.
        :param filename: str: The name of the file to save the instance.
        """
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

        logger.info(f'Saved SegmentPool instance to {filename}')

    def load(self, filename: str):
        """
        Load the SegmentPool instance from a file using pickle.
        :param filename: str: The name of the file to load the instance.
        """
        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)

        logger.info(f'Loaded SegmentPool instance from {filename}')

        logger.info(f'Set last length: {self.last_length}')
        logger.info(f'Set current max length: {self.current_max_length}')

    def export(self, filename: str, importance_scores: pd.DataFrame):
        """
        Save the segments table to a file.
        :param filename: str: The name of the file to save the lookup table.
        :param importance_scores: pd.DataFrame: The importance scores of the segments. id, importance
        """
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if filename.endswith('.txt'):
            with open(filename, 'w') as f:
                for item in self.segments:
                    f.write("%s\n" % item)
        elif filename.endswith('.csv'):
            if importance_scores is not None:
                # DataFrame with segments and their importance scores
                # pad the importance scores to match the length of segments

                # if i is in id, get the importance score. Else, 0
                scores = [importance_scores.get(i, 0) for i in range(len(self.segments))]

                df = pd.DataFrame({'Segment': self.segments, 'Importance': scores})
            else:
                # DataFrame with segments only
                df = pd.DataFrame({'Segment': self.segments})

            df.to_csv(filename, index=False)

        logger.info(f'Exported {len(self.segments)} segments to {filename}')

    def import_segments(self, filename: str):
        """
        Load the segments table from a file.
        :param filename: str: The name of the file to load the lookup table.
        """
        if filename.endswith('.txt'):
            with open(filename, 'r') as f:
                self.segments = f.read().splitlines()
        elif filename.endswith('.csv'):
            df = pd.read_csv(filename)
            self.segments = df['Segment'].tolist()

        logger.info(f'Imported {len(self.segments)} segments from {filename}')

    def add_subsequences(self, sequences: [str], current_length: int, remove_duplicates=True):
        """
        Add a list of sequences to the lookup table.
        Note: This method uses set to remove duplicates, but it changes the order of the sequences.
        :param sequences: list: The list of sequences to add.
        :param current_length: int: The current max length of the sequences.
        :param remove_duplicates: bool: Remove duplicates from the list if True.
        """
        self.segments = self.segments + sequences
        if remove_duplicates:
            self.segments = remove_duplicate_elements(self.segments)

        self.last_length = self.current_max_length
        self.current_max_length = current_length

        logger.info(f'Number of segments: {len(self.segments)}')

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

        eliminated = len(self.segments) - len(result)
        self.segments = result
        logger.info(f'Number of segments eliminated: {eliminated}')
        logger.info(f'Number of segments after redundant elimination: {len(self.segments)}')
        
    def n_gram_grafting(self):
        _ = [self._n_gram_grafting(i) for i in range(self.last_length, self.current_max_length + 1)]

    def _n_gram_grafting(self, n: int):
        """
        Combine two segments to if they match n-grams in the middle to create new segments.
        :param n: int: The length of the n-grams.
        """
        new_segments = [
            self.segments[i] + self.segments[j][n:]
            for i in range(len(self.segments)) if len(self.segments[i]) >= n
            for j in range(len(self.segments)) if
            len(self.segments[j]) >= n and i != j and len(self.segments[i]) + len(self.segments[j]) >= n
            if self.segments[i][-n:] == self.segments[j][:n]
        ]

        logger.info(f'Number of new segments after {n}-gram grafting: {len(new_segments)}')
        self.add_subsequences(new_segments, self.current_max_length)

    def fill(self, limit: int):
        """
        Fill the segments list with random sequences of the current length.
        :param limit: int: The number of segments to fill.
        """

        num_to_fill = limit - len(self.segments)
        if num_to_fill < 1:
            return

        logger.info(f'Filling {num_to_fill} segments')
        new_segments = [km.random_sequence(self.current_max_length) for _ in range(num_to_fill)]
        self.add_subsequences(new_segments, self.current_max_length)

    def get_current_max_length(self):
        return self.current_max_length
