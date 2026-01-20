# -*- coding: utf-8 -*-
# Author: Daniel Maddaleno
"""Pii Redactor — core implementation."""
"""PII detection and redaction validator."""

import re

from guardrails.pipeline import BaseValidator


class PIIRedactor(BaseValidator):
    """Detects and masks personally identifiable information."""

    PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "SSN": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
        "NAME": None,  # Handled separately with NER if available
    }

    def __init__(self, patterns: dict[str, str] | None = None, mask_char: str = "[{label}]"):
        self.patterns = patterns or {
            k: v for k, v in self.PATTERNS.items() if v is not None
        }
        self.mask_char = mask_char

    def validate(self, text: str) -> str:
        """Scan text for PII patterns and replace with labels."""
        masked = text
        for label, pattern in self.patterns.items():
            replacement = self.mask_char.format(label=label)
            masked = re.sub(pattern, replacement, masked)
        return masked

    def detect(self, text: str) -> list[dict]:
        """Return list of PII findings without masking."""
        findings = []
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                findings.append({
                    "type": label,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })
        return findings
