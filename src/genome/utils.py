def get_complement(sequence):
    """
    Get the complement of a sequence.
    :param sequence: str: The sequence.
    :return: str: The complement.
    """
    complement_map = {'a': 't', 't': 'a', 'g': 'c', 'c': 'g'}
    # reverse the sequence
    sequence = sequence[::-1]
    complement = "".join(complement_map[base] for base in sequence)
    return complement

def canonicalize(sequence):
    """
    Get the canonical kmer
    """
    return min(sequence, get_complement(sequence))