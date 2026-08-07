"""Unit tests for the secrets / credential detector."""

import pytest

from guardrails.pipeline import GuardrailViolation
from guardrails.validators.secrets import SecretsDetector


@pytest.fixture
def detector():
    return SecretsDetector()


class TestSecretsDetector:
    def test_flags_aws_access_key(self, detector):
        # The canonical AWS example key: AKIA + 16 uppercase/digit chars.
        text = "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        assert "aws_access_key_id" in detector.detect(text)

    def test_flags_github_token(self, detector):
        token = "ghp_" + "a" * 36
        assert "github_token" in detector.detect(f"token: {token}")

    def test_flags_private_key_header(self, detector):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA..."
        assert "private_key" in detector.detect(text)

    def test_validate_blocks_without_echoing_the_secret(self, detector):
        secret = "AKIAIOSFODNN7EXAMPLE"
        with pytest.raises(GuardrailViolation) as exc:
            detector.validate(f"here is the key {secret}")
        # The violation names the type but must not repeat the secret value,
        # otherwise the guard becomes a second leak.
        assert secret not in str(exc.value)
        assert "aws_access_key_id" in str(exc.value)

    def test_clean_text_passes_through(self, detector):
        text = "The quarterly report is ready for review."
        assert detector.detect(text) == []
        assert detector.validate(text) == text

    def test_word_in_prose_is_not_a_false_positive(self, detector):
        # "sk-" only starts a match on a word boundary, so ordinary hyphenated
        # words like "task-management" must not trip the openai_api_key rule.
        assert detector.detect("this is the task-management-overview doc") == []

    def test_extra_patterns_are_merged(self):
        detector = SecretsDetector(extra_patterns={"internal_token": r"\bTKN-[0-9]{6}\b"})
        found = detector.detect("use TKN-123456 to authenticate")
        assert "internal_token" in found

    def test_rejects_non_positive_threshold(self):
        # A threshold below 1 would make validate() block even clean text, since
        # an empty detection list still satisfies len(found) >= 0.
        with pytest.raises(ValueError):
            SecretsDetector(threshold=0)
        with pytest.raises(ValueError):
            SecretsDetector(threshold=-1)
