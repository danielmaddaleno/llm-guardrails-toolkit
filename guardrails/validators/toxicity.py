"""Toxicity / harmful-content detector using keyword heuristics."""

from __future__ import annotations

import re
import unicodedata

from guardrails.pipeline import BaseValidator, GuardrailViolation

# Invisible formatting characters an attacker can sprinkle inside a banned word
# ("raci<ZWSP>st") so a literal keyword regex misses it while a human still reads
# the slur normally. Same set the injection detector strips. In order: zero-width
# space, zero-width non-joiner, zero-width joiner, word joiner, BOM / zero-width
# no-break space, soft hyphen.
_INVISIBLE = dict.fromkeys(
    (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD),
    None,
)


def _normalize(text: str) -> str:
    """Fold obfuscation tricks away before keyword matching.

    Removes invisible characters and applies NFKC so compatibility forms such
    as fullwidth letters ("ｒａｃｉｓｔ") collapse to their ASCII equivalents. Used
    only for detection; the original text is what the pipeline passes on.
    """
    return unicodedata.normalize("NFKC", text.translate(_INVISIBLE))


# Lightweight keyword categories. Production systems should plug in
# a classifier model (e.g. OpenAI moderation, Perspective API).
_DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "hate_speech": [
        r"\bracis[tm]\b",
        r"\bsexis[tm]\b",
        r"\bhomophobi[ac]\b",
        r"\bxenophobi[ac]\b",
        r"\bwhite\s*supremac",
    ],
    "self_harm": [
        r"\bsuicid",
        r"\bself[- ]?harm",
    ],
    "violence": [
        r"\bkill\s+(him|her|them|you)",
        r"\bbomb\s+threat",
        r"\bmass\s+shoot",
    ],
}


class ToxicityDetector(BaseValidator):
    """Flag text that matches known harmful-content patterns.

    Parameters
    ----------
    categories : dict[str, list[str]] | None
        Mapping of category name -> list of regex patterns.
        Defaults to a small built-in set.  Replace with a model-based
        classifier for production workloads.
    threshold : int
        Number of distinct category matches before blocking. Default 1.
    """

    def __init__(
        self,
        categories: dict[str, list[str]] | None = None,
        threshold: int = 1,
    ):
        # A threshold below 1 makes validate() block every input: detect() can
        # return an empty list and ``len([]) >= 0`` is still true, so clean text
        # would trip the guard. Reject it up front instead of failing silently.
        if threshold < 1:
            raise ValueError(f"threshold must be at least 1, got {threshold}")
        self.categories = categories or _DEFAULT_CATEGORIES
        self.threshold = threshold
        self._compiled: dict[str, list[re.Pattern]] = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns] for cat, patterns in self.categories.items()
        }

    def validate(self, text: str) -> str:
        flagged = self.detect(text)
        if len(flagged) >= self.threshold:
            cats = ", ".join(flagged)
            raise GuardrailViolation(
                validator=self.name,
                message=f"Toxic content detected in categories: {cats}",
                severity="block",
            )
        return text

    def detect(self, text: str) -> list[str]:
        """Return list of matched category names.

        Matching runs against a normalized copy of the text so the same evasion
        tricks the injection detector already handles (invisible characters,
        fullwidth look-alikes) cannot slip a banned keyword past the filter.
        """
        normalized = _normalize(text)
        hits: list[str] = []
        for cat, patterns in self._compiled.items():
            if any(p.search(normalized) for p in patterns):
                hits.append(cat)
        return hits
