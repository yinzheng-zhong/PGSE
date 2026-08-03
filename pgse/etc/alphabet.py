"""
Alphabet configuration for PGSE.

PGSE was originally hard-wired to lowercase DNA ('a', 't', 'g', 'c'). This module
generalises that to an arbitrary set of characters with optional case sensitivity,
so the same segment-enhancement machinery can be applied to protein sequences,
natural language, or any other symbolic text.

A single :class:`Alphabet` instance is active at a time (see :func:`get_alphabet` /
:func:`set_alphabet`). The default is the original DNA alphabet, so existing code
and saved models keep behaving exactly as before.
"""

from typing import Iterable, Mapping, Optional, Union

DNA_CHARS = 'atgc'
DNA_COMPLEMENT = 'tacg'
DEFAULT_UNKNOWN_CHAR = 'n'

# Sentinel for "derive a sensible default from the other arguments".
AUTO = '__auto__'

# A complement can be given as a string parallel to the alphabet, a per-character
# mapping, ``None`` to disable canonicalisation, or the ``AUTO`` sentinel (a str).
ComplementArg = Union[str, Mapping[str, str], None]
# ``AUTO`` (a str), an explicit single character, or ``None`` to disable placeholders.
UnknownCharArg = Optional[str]
# Anything :func:`set_alphabet` will turn into an :class:`Alphabet`.
AlphabetArg = Union['Alphabet', str, Iterable[str], None]

# The byte tables handed to the C library: a 256-entry char -> index list, the
# alphabet size, and an optional 256-entry complement table.
ByteTables = tuple[list[int], int, Optional[bytes]]


class Alphabet:
    """
    The set of characters PGSE is allowed to see, plus how to treat them.

    :param chars: str or iterable of single characters: The allowed characters.
    :param case_sensitive: bool: If False (default), text and segments are lowercased
        and 'A' is treated as 'a'.
    :param complement: The complement mapping used for reverse-complement
        canonicalisation. Either a string parallel to ``chars``, a dict, ``None`` to
        disable canonicalisation, or ``AUTO`` to use the DNA mapping when ``chars``
        is the DNA alphabet and disable it otherwise.
    :param unknown_char: str or None: The placeholder substituted for characters
        outside the alphabet when read errors are kept. ``AUTO`` picks 'n' when it
        does not clash with the alphabet.
    """

    def __init__(
            self,
            chars: Union[str, Iterable[str]] = DNA_CHARS,
            case_sensitive: bool = False,
            complement: ComplementArg = AUTO,
            unknown_char: UnknownCharArg = AUTO
    ) -> None:
        given: list[str] = list(chars)
        if not given:
            raise ValueError('The alphabet must contain at least one character.')
        if any(len(c) != 1 for c in given):
            raise ValueError('Every entry of the alphabet must be a single character.')

        self.case_sensitive: bool = bool(case_sensitive)
        normalised: list[str] = given if self.case_sensitive else [c.lower() for c in given]
        if any(len(c) != 1 for c in normalised):
            raise ValueError(
                'Some characters change length when lower cased and cannot be used in a '
                'case-insensitive alphabet. Pass case_sensitive=True to keep them as they are.'
            )

        # Preserve the order the characters were given in. The order defines the
        # k-mer encoding, so it must be deterministic.
        self.chars: tuple[str, ...] = tuple(dict.fromkeys(normalised))
        self.char_set: frozenset[str] = frozenset(self.chars)

        self.unknown_char: Optional[str] = self._resolve_unknown_char(unknown_char)
        self.complement_map: Optional[dict[str, str]] = self._resolve_complement(
            complement, given, normalised
        )

        # Pre-computed lookups. They are built once here because they sit on hot paths.
        self._encoding: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self._encoding_with_unknown: dict[str, int] = dict(self._encoding)
        if self.unknown_char is not None:
            self._encoding_with_unknown[self.unknown_char] = len(self.chars)

        # A char -> char mapping as an ordinal table, ready for str.translate. Built
        # directly rather than via str.maketrans so the types stay concrete.
        self._translation: Optional[dict[int, int]] = (
            {ord(source): ord(target) for source, target in self.complement_map.items()}
            if self.complement_map else None
        )

    def __repr__(self) -> str:
        return (
            f"Alphabet(chars='{''.join(self.chars)}', case_sensitive={self.case_sensitive}, "
            f"complement={'on' if self.complement_map else 'off'}, unknown_char={self.unknown_char!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Alphabet):
            return NotImplemented
        return (
            self.chars == other.chars
            and self.case_sensitive == other.case_sensitive
            and self.complement_map == other.complement_map
            and self.unknown_char == other.unknown_char
        )

    def __hash__(self) -> int:
        return hash((self.chars, self.case_sensitive, self.unknown_char, bool(self.complement_map)))

    def _resolve_unknown_char(self, unknown_char: UnknownCharArg) -> Optional[str]:
        if unknown_char == AUTO:
            return DEFAULT_UNKNOWN_CHAR if DEFAULT_UNKNOWN_CHAR not in self.char_set else None
        if unknown_char is None:
            return None
        if len(unknown_char) != 1:
            raise ValueError('unknown_char must be a single character.')
        if not self.case_sensitive:
            unknown_char = unknown_char.lower()
        if unknown_char in self.char_set:
            raise ValueError(
                f"unknown_char '{unknown_char}' is part of the alphabet and cannot be used as a placeholder."
            )
        return unknown_char

    def _resolve_complement(
            self,
            complement: ComplementArg,
            given: list[str],
            normalised: list[str]
    ) -> Optional[dict[str, str]]:
        if isinstance(complement, str) and complement == AUTO:
            if self.char_set != frozenset(DNA_CHARS):
                return None
            complement = dict(zip(DNA_CHARS, DNA_COMPLEMENT))
        if not complement:  # None or ''
            return None

        pairs: list[tuple[str, str]]
        if isinstance(complement, Mapping):
            pairs = list(complement.items())
        else:
            complement_chars = list(complement)
            if len(complement_chars) != len(given):
                raise ValueError(
                    f'The complement must have one character per alphabet character '
                    f'({len(given)} expected, {len(complement_chars)} given).'
                )
            pairs = list(zip(normalised, complement_chars))

        mapping: dict[str, str] = {}
        for source, target in pairs:
            if len(source) != 1 or len(target) != 1:
                raise ValueError('The complement must map single characters to single characters.')
            if not self.case_sensitive:
                source, target = source.lower(), target.lower()
            if mapping.setdefault(source, target) != target:
                raise ValueError(f"Conflicting complements given for '{source}'.")

        unknown = (set(mapping) | set(mapping.values())) - self.char_set
        if self.unknown_char is not None:
            unknown -= {self.unknown_char}
        if unknown:
            raise ValueError(f'The complement refers to characters outside the alphabet: {sorted(unknown)}')

        # Complementing is symmetric, so a pair only has to be given once.
        for source, target in list(mapping.items()):
            mapping.setdefault(target, source)

        # Characters without an explicit complement (e.g. the unknown placeholder)
        # map to themselves, matching the original DNA behaviour.
        for char in self.chars:
            mapping.setdefault(char, char)
        if self.unknown_char is not None:
            mapping.setdefault(self.unknown_char, self.unknown_char)

        for source, target in mapping.items():
            if mapping[target] != source:
                raise ValueError(
                    f"The complement is not reversible: '{source}' -> '{target}' -> '{mapping[target]}'."
                )

        return mapping

    @property
    def size(self) -> int:
        """Number of characters in the alphabet, excluding the unknown placeholder."""
        return len(self.chars)

    @property
    def has_complement(self) -> bool:
        """True when the alphabet supports reverse-complement canonicalisation."""
        return self._translation is not None

    @property
    def is_byte_safe(self) -> bool:
        """True when every character fits in a single byte, i.e. the C library can be used."""
        return all(ord(c) < 256 for c in self.chars) and (
            self.unknown_char is None or ord(self.unknown_char) < 256
        )

    def encoding_map(self, include_unknown: bool = False) -> dict[str, int]:
        """
        Map each character to its integer code, used for k-mer encoding.

        :param include_unknown: bool: Include the unknown placeholder in the map.
        :return: dict: The character to index map.
        """
        if include_unknown:
            if self.unknown_char is None:
                raise ValueError(
                    f'{self} has no unknown placeholder, so read errors cannot be kept. '
                    f'Pass unknown_char to the alphabet to enable it.'
                )
            return self._encoding_with_unknown
        return self._encoding

    def base(self, include_unknown: bool = False) -> int:
        """Size of the k-mer encoding base."""
        return len(self.encoding_map(include_unknown))

    def characters(self, include_unknown: bool = False) -> list[str]:
        """The characters used to build segments, optionally including the placeholder."""
        return list(self.encoding_map(include_unknown).keys())

    def normalise(self, text: str) -> str:
        """Apply case folding when the alphabet is case-insensitive."""
        return text if self.case_sensitive else text.lower()

    # American spelling kept as an alias since the rest of the code base uses it.
    normalize = normalise

    def sanitise(self, text: str, keep_read_error: bool = False) -> str:
        """
        Normalise the case and drop (or replace) every character outside the alphabet.

        :param text: str: The text to sanitise.
        :param keep_read_error: bool: Replace unexpected characters with the unknown
            placeholder instead of removing them.
        :return: str: The sanitised text.
        """
        text = self.normalise(text)
        allowed = self.char_set

        if keep_read_error:
            unknown = self.unknown_char
            if unknown is None:
                raise ValueError(
                    f'{self} has no unknown placeholder, so read errors cannot be kept. '
                    f'Pass unknown_char to the alphabet to enable it.'
                )
            return ''.join([c if c in allowed else unknown for c in text])

        return ''.join([c for c in text if c in allowed])

    sanitize = sanitise

    def get_complement(self, sequence: str) -> str:
        """
        Reverse complement a sequence. Returns the sequence unchanged when the
        alphabet has no complement, which turns canonicalisation into a no-op.

        :param sequence: str: The sequence.
        :return: str: The reverse complement.
        """
        if self._translation is None:
            return sequence
        return sequence.translate(self._translation)[::-1]

    def canonicalize(self, sequence: str) -> str:
        """Get the canonical form of a sequence."""
        return min(sequence, self.get_complement(sequence))

    canonicalise = canonicalize

    def byte_tables(self) -> ByteTables:
        """
        Build the byte lookup tables consumed by the Aho-Corasick C library.

        :return: tuple: (list of 256 indices, alphabet size, complement bytes or None).
            Characters outside the alphabet map to -1. When the alphabet is
            case-insensitive both cases map to the same index. The unknown placeholder
            gets an index of its own, so that segments carrying read errors can match
            just as they do in the Python implementation.
        """
        if not self.is_byte_safe:
            raise ValueError(f'{self} contains multi-byte characters and cannot be used with the C library.')

        index: list[int] = [-1] * 256
        complement = bytearray(range(256))
        characters = self.characters(include_unknown=self.unknown_char is not None)

        def byte_of(char: str) -> Optional[int]:
            # Upper casing can change the length of a character ('ß' -> 'SS'), in which
            # case there is no single byte to index and the variant is skipped.
            return ord(char) if len(char) == 1 and ord(char) < 256 else None

        for position, char in enumerate(characters):
            variants = {char} if self.case_sensitive else {char.lower(), char.upper()}
            for variant in variants:
                byte = byte_of(variant)
                if byte is not None:
                    index[byte] = position

        if self.complement_map:
            for source, target in self.complement_map.items():
                pair_variants = [(source, target)] if self.case_sensitive else [
                    (source.lower(), target.lower()), (source.upper(), target.upper())
                ]
                for variant_source, variant_target in pair_variants:
                    source_byte, target_byte = byte_of(variant_source), byte_of(variant_target)
                    if source_byte is not None and target_byte is not None:
                        complement[source_byte] = target_byte

        return index, len(characters), (bytes(complement) if self.complement_map else None)


# The DNA alphabet PGSE has always used. Kept as the default so nothing changes
# for existing callers.
DNA = Alphabet()

_active_alphabet: Alphabet = DNA


def get_alphabet() -> Alphabet:
    """
    Get the alphabet currently in use.

    :return: Alphabet: The active alphabet.
    """
    return _active_alphabet


def set_alphabet(
        alphabet: AlphabetArg = None,
        case_sensitive: bool = False,
        complement: ComplementArg = AUTO,
        unknown_char: UnknownCharArg = AUTO
) -> Alphabet:
    """
    Set the alphabet used by the whole package.

    Accepts either a ready-made :class:`Alphabet` or the arguments to build one::

        set_alphabet('abcdefghijklmnopqrstuvwxyz ')
        set_alphabet('atgcATGC', case_sensitive=True)
        set_alphabet(Alphabet('atgc', complement='tacg'))

    This must be called before sequences are loaded, and it is propagated to the
    Ray workers by the loaders.

    :return: Alphabet: The alphabet that was installed.
    """
    global _active_alphabet

    if alphabet is None:
        alphabet = DNA_CHARS
    if not isinstance(alphabet, Alphabet):
        alphabet = Alphabet(
            alphabet,
            case_sensitive=case_sensitive,
            complement=complement,
            unknown_char=unknown_char
        )

    _active_alphabet = alphabet
    return alphabet


def reset_alphabet() -> Alphabet:
    """Restore the default DNA alphabet."""
    return set_alphabet(DNA)


def get_complement(sequence: str) -> str:
    """
    Get the complement of a sequence using the active alphabet.

    :param sequence: str: The sequence.
    :return: str: The complement.
    """
    return _active_alphabet.get_complement(sequence)


def canonicalize(sequence: str) -> str:
    """
    Get the canonical kmer using the active alphabet.
    """
    return _active_alphabet.canonicalize(sequence)


def is_canonical(seq: str, complement_seq: str) -> bool:
    return seq <= complement_seq
