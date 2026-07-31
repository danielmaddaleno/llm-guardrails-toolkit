"""PII detection and redaction validator."""

from __future__ import annotations

import re
from typing import Callable

from guardrails.pipeline import BaseValidator


def _luhn_valid(number: str) -> bool:
    """Return True if the digits of *number* pass the Luhn checksum.

    Real credit-card numbers satisfy the Luhn (mod 10) check by construction,
    so requiring it lets us drop 16-digit strings that only look like a card
    (order ids, tracking numbers, concatenated timestamps) without ever
    dropping a genuine card. Non-digit separators are ignored before the check.
    """
    digits = [int(c) for c in re.sub(r"\D", "", number)]
    if len(digits) < 2:
        return False
    total = 0
    # Double every second digit counting from the right, per the Luhn rule.
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class PIIRedactor(BaseValidator):
    """Detects and masks personally identifiable information."""

    PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "SSN": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
        "NAME": None,  # Handled separately with NER if available
    }

    # Optional per-label confirmation checks applied to each raw regex match.
    # A label with no entry is always accepted; a confirm that returns False
    # drops the match. This is how a 16-digit string that fails the Luhn check
    # is kept out of both the redaction and the detect() findings.
    CONFIRMERS: dict[str, Callable[[str], bool]] = {
        "CREDIT_CARD": _luhn_valid,
    }

    def __init__(self, patterns: dict[str, str] | None = None, mask_char: str = "[{label}]"):
        self.patterns = patterns or {k: v for k, v in self.PATTERNS.items() if v is not None}
        self.mask_char = mask_char

    def validate(self, text: str) -> str:
        """Scan text for PII patterns and replace each finding with its label.

        Redaction is driven by the same span search as :meth:`detect`, so the
        two can never disagree, and any overlapping matches are collapsed to a
        single span: a credit-card number is masked once as ``[CREDIT_CARD]``
        rather than partly re-masked by a shorter overlapping rule.
        """
        findings = self.detect(text)
        if not findings:
            return text

        out: list[str] = []
        cursor = 0
        for finding in findings:
            start, end = finding["start"], finding["end"]
            out.append(text[cursor:start])
            out.append(self.mask_char.format(label=finding["type"]))
            cursor = end
        out.append(text[cursor:])
        return "".join(out)

    def detect(self, text: str) -> list[dict]:
        """Return non-overlapping PII findings in reading order.

        Each pattern is searched independently; matches that fail their label's
        confirmation check (for example the Luhn test for credit cards) are
        dropped. The survivors are then reduced to a non-overlapping set with a
        leftmost-longest rule and returned sorted by position, which keeps the
        result usable both for masking and for building an audit log.
        """
        candidates: list[dict] = []
        for label, pattern in self.patterns.items():
            confirm = self.CONFIRMERS.get(label)
            for match in re.finditer(pattern, text):
                if confirm is not None and not confirm(match.group()):
                    continue
                candidates.append(
                    {
                        "type": label,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        # Leftmost-longest wins: sort by start, then by longer span first, and
        # keep a match only if it starts at or after the end of the last kept
        # one. This stops a shorter rule (e.g. SSN) from carving up a longer
        # match (e.g. a credit-card number) that happens to overlap it.
        candidates.sort(key=lambda f: (f["start"], -(f["end"] - f["start"])))
        findings: list[dict] = []
        last_end = -1
        for candidate in candidates:
            if candidate["start"] >= last_end:
                findings.append(candidate)
                last_end = candidate["end"]
        return findings
