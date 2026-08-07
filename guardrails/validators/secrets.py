"""Secret / credential leak detection validator."""

from __future__ import annotations

import re

from guardrails.pipeline import BaseValidator, GuardrailViolation


class SecretsDetector(BaseValidator):
    """Block text that contains what looks like a live credential.

    Aimed at prompts and, more importantly, model output: an LLM that has seen
    a key in its context can echo it straight back to a user. This guard targets
    structured, prefixed secrets (AWS keys, GitHub/Slack tokens, Google/OpenAI
    API keys, PEM private keys) rather than high-entropy heuristics, which keeps
    false positives low at the cost of missing unprefixed or custom tokens. Pair
    it with a real scanner (detect-secrets, TruffleHog) for anything critical.

    The matched value is never echoed back in the violation message, so the
    guard does not turn into a second place the secret leaks.
    """

    # Patterns are case-sensitive on purpose: real credential formats have fixed
    # casing (AKIA..., ghp_..., AIza...), so matching case-insensitively would
    # only widen the net and let in more false positives.
    DEFAULT_PATTERNS: dict[str, str] = {
        "aws_access_key_id": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        "github_token": r"\bgh[oprsu]_[A-Za-z0-9]{36,}\b",
        "slack_token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        "google_api_key": r"\bAIza[0-9A-Za-z_-]{35}\b",
        "openai_api_key": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b",
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
    }

    def __init__(
        self,
        extra_patterns: dict[str, str] | None = None,
        threshold: int = 1,
    ):
        # A threshold below 1 makes validate() block every input: detect() can
        # return an empty list and ``len([]) >= 0`` is still true, so clean text
        # would trip the guard. Reject it up front instead of failing silently.
        if threshold < 1:
            raise ValueError(f"threshold must be at least 1, got {threshold}")
        patterns = {**self.DEFAULT_PATTERNS, **(extra_patterns or {})}
        self.threshold = threshold
        self._compiled: dict[str, re.Pattern] = {name: re.compile(p) for name, p in patterns.items()}

    def validate(self, text: str) -> str:
        """Raise if the text carries at least ``threshold`` kinds of secret."""
        found = self.detect(text)
        if len(found) >= self.threshold:
            raise GuardrailViolation(
                validator=self.name,
                # Name the kinds of secret, never the matched values, so the
                # error message does not leak the credential a second time.
                message=f"Detected potential secret(s): {found}",
                severity="block",
            )
        return text

    def detect(self, text: str) -> list[str]:
        """Return the names of the secret types found in the text."""
        return [name for name, pattern in self._compiled.items() if pattern.search(text)]
