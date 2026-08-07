"""Helpers for activating an alphabet temporarily and for describing one."""

from contextlib import contextmanager
from typing import Any, Iterator

from pgse.dataset.alphabet import Alphabet, get_alphabet, set_alphabet


@contextmanager
def using_alphabet(alphabet: Alphabet) -> Iterator[Alphabet]:
    """Make an alphabet active for the duration of the block, then restore the previous one.

    Args:
        alphabet: The alphabet to activate.
    """
    previous = get_alphabet()
    set_alphabet(alphabet)
    try:
        yield alphabet
    finally:
        set_alphabet(previous)


def alphabet_to_dict(alphabet: Alphabet) -> dict[str, Any]:
    """Describe an alphabet with JSON-serialisable values.

    Args:
        alphabet: The alphabet to describe.
    """
    return {
        'chars': ''.join(alphabet.chars),
        'case_sensitive': alphabet.case_sensitive,
        'complement': alphabet.complement_map,
        'unknown_char': alphabet.unknown_char,
    }


def alphabet_from_dict(description: dict[str, Any]) -> Alphabet:
    """Rebuild the alphabet described by alphabet_to_dict.

    Args:
        description: The description to read.
    """
    return Alphabet(
        description['chars'],
        case_sensitive=bool(description.get('case_sensitive', False)),
        complement=description.get('complement'),
        unknown_char=description.get('unknown_char'),
    )
