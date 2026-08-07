from typing import Optional

from pgse.dataset.alphabet import canonicalize, get_complement, is_canonical

__all__ = ["canonicalize", "get_complement", "is_canonical", "parse_fasta"]


def parse_fasta(text: str) -> list[str]:
    """Split FASTA text into its contigs, reading header-less text as a single contig.

    Args:
        text: The contents of a FASTA file, or a bare sequence.
    """
    lines = text.split('\n')
    headers = [i for i, line in enumerate(lines) if line.startswith('>')]

    if not headers:
        return [''.join(lines)]

    ends: list[Optional[int]] = list(headers[1:]) + [None]
    return [''.join(lines[start + 1:end]) for start, end in zip(headers, ends)]
