"""Text normalization shared by the pattern and keyword detectors."""

from __future__ import annotations

import unicodedata


def normalize_for_matching(text: str) -> str:
    """Fold obfuscation tricks away before a detector matches against text.

    Drops every Unicode Format (Cf) character. That category is exactly the
    set of zero-width and bidirectional control codes an attacker wedges
    inside a banned phrase (a zero-width space in "ig...nore", a left-to-right
    mark in "raci...st") to slip it past a literal regex while the text still
    reads normally to a human. Matching the whole category, rather than a
    hand-listed set of code points, keeps a newly discovered invisible
    character from reopening the gap in one detector but not the other.

    Then applies NFKC so compatibility forms such as fullwidth Latin letters
    collapse to their ASCII equivalents. Used for detection only; the pipeline
    still passes the original text on unchanged.
    """
    stripped = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return unicodedata.normalize("NFKC", stripped)
