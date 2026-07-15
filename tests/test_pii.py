"""Unit tests for PII redaction validator."""

import pytest

from guardrails.validators.pii_redactor import PIIRedactor


@pytest.fixture
def redactor():
    return PIIRedactor()


class TestPIIRedactor:
    def test_redacts_email(self, redactor):
        text = "Contact me at john.doe@example.com please."
        result = redactor.validate(text)
        assert "john.doe@example.com" not in result
        assert "[EMAIL]" in result

    def test_redacts_phone(self, redactor):
        text = "Call me at 555-123-4567."
        result = redactor.validate(text)
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    def test_redacts_ssn(self, redactor):
        text = "My SSN is 123-45-6789."
        result = redactor.validate(text)
        assert "123-45-6789" not in result
        assert "[SSN]" in result

    def test_redacts_credit_card(self, redactor):
        text = "Card number: 4111-1111-1111-1111"
        result = redactor.validate(text)
        assert "4111-1111-1111-1111" not in result
        assert "[CREDIT_CARD]" in result

    def test_no_pii_passthrough(self, redactor):
        text = "This is a perfectly safe sentence."
        result = redactor.validate(text)
        assert result == text

    def test_multiple_pii(self, redactor, text_with_pii):
        result = redactor.validate(text_with_pii)
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "[SSN]" in result
        assert "john.doe@example.com" not in result

    def test_detect_returns_types(self, redactor):
        text = "john@test.com and 111-22-3333"
        detected = redactor.detect(text)
        types = {d["type"] for d in detected}
        assert "EMAIL" in types
        assert "SSN" in types
