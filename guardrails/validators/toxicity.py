# -*- coding: utf-8 -*-
"""Toxicity / harmful-content detector using keyword heuristics."""

import re
from guardrails.pipeline import BaseValidator, GuardrailViolation

# Lightweight keyword categories – production systems should plug in
# a classifier model (e.g. OpenAI moderation, Perspective API).
_DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "hate_speech": [
        r"\bracis[tm]\b", r"\bsexis[tm]\b", r"\bhomophobi[ac]\b",
        r"\bxenophobi[ac]\b", r"\bwhite\s*supremac",
    ],
    "self_harm": [
        r"\bsuicid", r"\bself[- ]?harm",
    ],
    "violence": [
        r"\bkill\s+(him|her|them|you)", r"\bbomb\s+threat",
        r"\bmass\s+shoot",
    ],
}


class ToxicityDetector(BaseValidator):
    """Flag text that matches known harmful-content patterns.

    Parameters
    ----------
    categories : dict[str, list[str]] | None
        Mapping of category name → list of regex patterns.
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
        self.categories = categories or _DEFAULT_CATEGORIES
        self.threshold = threshold
        self._compiled: dict[str, list[re.Pattern]] = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.categories.items()
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
        """Return list of matched category names."""
        hits: list[str] = []
        for cat, patterns in self._compiled.items():
            if any(p.search(text) for p in patterns):
                hits.append(cat)
        return hits
