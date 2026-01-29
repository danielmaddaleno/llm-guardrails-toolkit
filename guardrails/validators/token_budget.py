# -*- coding: utf-8 -*-
# Author: Daniel Maddaleno
"""Token Budget — core implementation."""
"""Token budget enforcement validator."""

from guardrails.pipeline import BaseValidator, GuardrailViolation


class TokenBudget(BaseValidator):
    """Enforce token limits on input or output text.

    Uses a simple word-based approximation (1 token ≈ 0.75 words)
    for fast estimation without requiring a tokenizer dependency.
    """

    def __init__(self, max_tokens: int = 4000, chars_per_token: float = 4.0):
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    def validate(self, text: str) -> str:
        estimated = self.estimate_tokens(text)
        if estimated > self.max_tokens:
            raise GuardrailViolation(
                validator=self.name,
                message=f"Estimated {estimated} tokens exceeds limit of {self.max_tokens}",
                severity="block",
            )
        return text

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from character length."""
        return int(len(text) / self.chars_per_token)
