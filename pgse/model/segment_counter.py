"""Segment counting for a fixed set of segments, independent of the global pool."""

from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt

from pgse.algos import aho_corasick, native_counter
from pgse.dataset.alphabet import Alphabet
from pgse.dataset.alphabet_utils import using_alphabet
from pgse.dataset.counts import Dataset, assemble_counts

COUNTING_DESCRIPTION = 'Counting segments'


class SegmentCounter:
    """Counts its own segments against sequences, without Ray and without the global pool."""

    def __init__(
            self,
            segments: Sequence[str],
            alphabet: Alphabet,
            count_dtype: npt.DTypeLike = np.float32,
            sparse: bool = False,
            threads: int = 8
    ) -> None:
        """
        Args:
            segments: The segments to count, ordered as the columns of the count matrix.
            alphabet: The alphabet the segments were built with.
            count_dtype: Storage dtype of the counts (np.float32 or np.uint16).
            sparse: Store the counts as a sparse CSR matrix instead of a dense array.
            threads: Threads the native counting kernel may use.
        """
        self.segments: list[str] = list(segments)
        self.alphabet: Alphabet = alphabet
        self.count_dtype: npt.DTypeLike = count_dtype
        self.sparse: bool = sparse
        self.threads: int = max(threads, 1)

        self._matcher: Optional[Any] = None

    def count(self, sequences: Sequence[Any]) -> Dataset:
        """Build the count matrix of the sequences, one row each.

        Args:
            sequences: The Sequence objects to count against.
        """
        with using_alphabet(self.alphabet):
            if native_counter.native_available():
                return native_counter.count_matrix(
                    self._get_matcher(), sequences, len(self.segments),
                    dtype=self.count_dtype, sparse=self.sparse,
                    threads=self.threads, desc=COUNTING_DESCRIPTION
                )

            rows = (
                np.asarray(aho_corasick.count_segments(sequence.nodes, self.segments))
                for sequence in sequences
            )
            return assemble_counts(
                rows, len(sequences), len(self.segments),
                dtype=self.count_dtype, sparse=self.sparse, desc=COUNTING_DESCRIPTION
            )

    def _get_matcher(self) -> Any:
        """The Aho-Corasick automaton over the segments, built once and reused."""
        if self._matcher is None:
            self._matcher = native_counter.build_matcher(self.segments, self.alphabet)

        return self._matcher
