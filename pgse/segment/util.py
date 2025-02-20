def remove_duplicate_elements(lst):
    """
    Remove duplicate elements from a list. Makes sure the output list will always be the same if the input list is the same.
    """
    return list(dict.fromkeys(lst))