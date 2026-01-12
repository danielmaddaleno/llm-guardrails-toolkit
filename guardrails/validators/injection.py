# -*- coding: utf-8 -*-
# Author: Daniel Maddaleno
"""Injection — core implementation."""
"""Prompt injection detection validator."""

import re

from guardrails.pipeline import BaseValidator, GuardrailViolation


class PromptInjectionDetector(BaseValidator):
    """Detects common prompt injection patterns and blocks them."""

    DEFAULT_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(everything|all|your)\s+(above|previous|instructions)",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"new\s+instructions?\s*:",
        r"system\s*prompt\s*:",
        r"act\s+as\s+(if\s+)?you\s+(are|were)",
        r"pretend\s+(you\s+are|to\s+be)",
        r"do\s+not\s+follow\s+your\s+(rules|guidelines|instructions)",
        r"override\s+(your\s+)?(rules|system|instructions)",
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
        """Return list of matched injection patterns."""
        matches = []
        for pattern in self.patterns:
            if re.search(pattern, text, self.flags):
                matches.append(pattern)
        return matches
