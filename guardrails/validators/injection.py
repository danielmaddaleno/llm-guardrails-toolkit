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
    """Detects common prompt injection patterns.

    Patterns sit in one of two tiers. ``BLOCKING_PATTERNS`` are phrases that
    have no ordinary use in a user prompt, so a match raises a block-severity
    violation. ``ADVISORY_PATTERNS`` are phrases that attackers use but honest
    users also write ("act as if you were a reviewer", a "### Instructions"
    heading), so a match raises a warn-severity violation instead: it shows up
    in ``ValidationResult.violations`` for auditing while ``is_safe`` stays
    True. Keeping them in the blocking tier flagged about a third of the benign
    prompts in the test corpus, which is the kind of rate that gets a guard
    switched off entirely.
    """

    BLOCKING_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?|directions?|rules|messages?)",
        r"forget\s+(everything|all|your)\s+(above|previous|instructions)",
        r"(show|reveal|print|output)\s+(me\s+)?(the\s+)?system\s*prompt",
        r"do\s+not\s+follow\s+your\s+(rules|guidelines|instructions)",
        r"override\s+your\s+(rules|system|instructions)",
        r"\[system\]",
        r"<\|im_start\|>",
    ]

    ADVISORY_PATTERNS = [
        r"disregard\s+(all\s+)?(previous|prior)",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"new\s+instructions?\s*:",
        r"system\s*prompt\s*:",
        r"act\s+as\s+(if\s+)?you\s+(are|were)",
        r"pretend\s+(you\s+are|to\s+be)",
        r"override\s+(the\s+)?(rules|system|instructions)",
        r"jailbreak",
        r"###\s*(system|instruction)",
    ]

    def __init__(
        self,
        extra_patterns: list[str] | None = None,
        extra_advisory_patterns: list[str] | None = None,
        case_sensitive: bool = False,
        threshold: int = 1,
    ):
        # A threshold below 1 makes validate() block every input: detect() can
        # return an empty list and ``len([]) >= 0`` is still true, so clean text
        # would trip the guard. Reject it up front instead of failing silently.
        if threshold < 1:
            raise ValueError(f"threshold must be at least 1, got {threshold}")
        self.patterns = self.BLOCKING_PATTERNS + (extra_patterns or [])
        self.advisory_patterns = self.ADVISORY_PATTERNS + (extra_advisory_patterns or [])
        self.flags = 0 if case_sensitive else re.IGNORECASE
        self.threshold = threshold
        # Compile once at construction so we don't rebuild every pattern on
        # each validate() call, which happens on the hot path for every
        # request. Mirrors how ToxicityDetector handles its patterns.
        self._compiled = [re.compile(p, self.flags) for p in self.patterns]
        self._compiled_advisory = [re.compile(p, self.flags) for p in self.advisory_patterns]

    def validate(self, text: str) -> str:
        """Check for prompt injection patterns.

        Raises:
            GuardrailViolation: severity ``"block"`` once at least ``threshold``
                blocking patterns match, or severity ``"warn"`` when only
                advisory patterns do.
        """
        normalized = _normalize(text)
        blocking = self._matching(self._compiled, normalized)
        if len(blocking) >= self.threshold:
            raise GuardrailViolation(
                validator=self.name,
                message=f"Detected {len(blocking)} injection pattern(s): {blocking}",
                severity="block",
            )

        advisory = self._matching(self._compiled_advisory, normalized)
        if advisory:
            raise GuardrailViolation(
                validator=self.name,
                message=f"Detected {len(advisory)} ambiguous pattern(s): {advisory}",
                severity="warn",
            )
        return text

    def detect(self, text: str) -> list[str]:
        """Return every matched pattern, blocking tier first."""
        return self.detect_blocking(text) + self.detect_advisory(text)

    def detect_blocking(self, text: str) -> list[str]:
        """Return the matched patterns that are worth blocking on."""
        return self._matching(self._compiled, _normalize(text))

    def detect_advisory(self, text: str) -> list[str]:
        """Return the matched patterns that only warrant a warning."""
        return self._matching(self._compiled_advisory, _normalize(text))

    @staticmethod
    def _matching(compiled: list[re.Pattern], normalized: str) -> list[str]:
        """Match against normalized text so evasion tricks do not hide a pattern."""
        return [pattern.pattern for pattern in compiled if pattern.search(normalized)]
