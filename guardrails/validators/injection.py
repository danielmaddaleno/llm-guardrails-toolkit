"""Prompt injection detection validator."""

from __future__ import annotations

import re
import unicodedata

from guardrails.pipeline import BaseValidator, GuardrailViolation

# Zero-width and other invisible formatting characters. An attacker can drop
# these between letters ("ig<ZWSP>nore previous instructions") to slip a known
# phrase past a literal regex while the text still reads normally to a human.
# We strip them before matching. In order: zero-width space, zero-width
# non-joiner, zero-width joiner, word joiner, BOM / zero-width no-break space,
# soft hyphen.
_INVISIBLE = dict.fromkeys(
    (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD),
    None,
)


def _normalize(text: str) -> str:
    """Fold obfuscation tricks away before pattern matching.

    Removes invisible characters and applies NFKC so compatibility forms such
    as fullwidth letters ("ｉｇｎｏｒｅ") collapse to their ASCII equivalents. Used
    only for detection; the original text is what the pipeline passes on.
    """
    return unicodedata.normalize("NFKC", text.translate(_INVISIBLE))


class PromptInjectionDetector(BaseValidator):
    """Detects common prompt injection patterns and blocks them."""

    DEFAULT_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?(previous|prior)",
        r"forget\s+(everything|all|your)\s+(above|previous|instructions)",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"new\s+instructions?\s*:",
        r"(show|reveal|print|output)\s+(me\s+)?(the\s+)?system\s*prompt",
        r"system\s*prompt\s*:",
        r"act\s+as\s+(if\s+)?you\s+(are|were)",
        r"pretend\s+(you\s+are|to\s+be)",
        r"do\s+not\s+follow\s+your\s+(rules|guidelines|instructions)",
        r"override\s+(your\s+)?(rules|system|instructions)",
        r"jailbreak",
        r"\[system\]",
        r"<\|im_start\|>",
        r"###\s*(system|instruction)",
    ]

    def __init__(
        self,
        extra_patterns: list[str] | None = None,
        case_sensitive: bool = False,
        threshold: int = 1,
    ):
        self.patterns = self.DEFAULT_PATTERNS + (extra_patterns or [])
        self.flags = 0 if case_sensitive else re.IGNORECASE
        self.threshold = threshold
        # Compile once at construction so we don't rebuild every pattern on
        # each validate() call, which happens on the hot path for every
        # request. Mirrors how ToxicityDetector handles its patterns.
        self._compiled = [re.compile(p, self.flags) for p in self.patterns]

    def validate(self, text: str) -> str:
        """Check for prompt injection patterns.

        Raises:
            GuardrailViolation: If injection patterns are detected.
        """
        detections = self.detect(text)
        if len(detections) >= self.threshold:
            raise GuardrailViolation(
                validator=self.name,
                message=f"Detected {len(detections)} injection pattern(s): {detections}",
                severity="block",
            )
        return text

    def detect(self, text: str) -> list[str]:
        """Return list of matched injection patterns.

        Matching runs against a normalized copy of the text so common evasion
        tricks (invisible characters, fullwidth look-alikes) do not hide an
        otherwise obvious pattern.
        """
        normalized = _normalize(text)
        return [pattern.pattern for pattern in self._compiled if pattern.search(normalized)]
