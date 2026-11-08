"""Text normalization shared by the pattern and keyword detectors."""

from __future__ import annotations

import unicodedata


def strip_format_chars(text: str) -> tuple[str, list[int]]:
    """Drop Unicode Format (Cf) characters, keeping a map back to the original.

    The Cf category is exactly the set of zero-width and bidirectional control
    codes an attacker wedges inside a value (a zero-width space in "ig...nore",
    a left-to-right mark between two digits of a card number) to slip it past a
    literal regex while the text still reads normally to a human.

    Returns the text with those characters removed and a list whose i-th entry
    is the index in *text* of the i-th surviving character. A detector that
    needs spans in the original string, as the PII redactor does when it masks
    in place, runs its regex on the cleaned text and maps each match back
    through this list, so a value split by an invisible character is still
    found and masked over its original span.
    """
    cleaned: list[str] = []
    index_map: list[int] = []
    for index, ch in enumerate(text):
        if unicodedata.category(ch) == "Cf":
            continue
        cleaned.append(ch)
        index_map.append(index)
    return "".join(cleaned), index_map


def normalize_for_matching(text: str) -> str:
    """Fold obfuscation tricks away before a detector matches against text.

    Drops every Unicode Format (Cf) character (see :func:`strip_format_chars`)
    so a wedged zero-width or bidi mark cannot break a literal match, then
    applies NFKC so compatibility forms such as fullwidth Latin letters
    collapse to their ASCII equivalents. Used for detection only; the pipeline
    still passes the original text on unchanged. The index map is not needed
    here (the result is matched as a whole, not mapped back to spans), so it is
    discarded.
    """
    stripped, _ = strip_format_chars(text)
    return unicodedata.normalize("NFKC", stripped)
