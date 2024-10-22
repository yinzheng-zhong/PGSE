def get_complement(sequence):
    """
    Get the complement of a sequence.
    :param sequence: str: The sequence.
    :return: str: The complement.
    """
    # Create the translation table once outside the function
    complement_map = str.maketrans('atgc', 'tacg')

    # Translate and reverse in one step
    return sequence.translate(complement_map)[::-1]


def canonicalize(sequence):
    """
    Get the canonical kmer
    """
    return min(sequence, get_complement(sequence))