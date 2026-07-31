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

    def test_detect_orders_by_position(self, redactor):
        # The SSN appears before the email in the text, but EMAIL is scanned
        # first, so without sorting the email would come back first. detect()
        # should return findings in the order they appear in the text.
        text = "SSN 111-22-3333, reach me at john@test.com"
        detected = redactor.detect(text)
        starts = [d["start"] for d in detected]
        assert starts == sorted(starts)
        assert [d["type"] for d in detected] == ["SSN", "EMAIL"]

    def test_credit_card_failing_luhn_is_not_redacted(self, redactor):
        # 4111-1111-1111-1111 is a valid Luhn card; flipping the last digit
        # breaks the checksum, so this 16-digit string is not a real card and
        # must survive untouched instead of being masked as [CREDIT_CARD].
        text = "Reference number 4111-1111-1111-1112 for the order."
        assert redactor.validate(text) == text
        assert not any(d["type"] == "CREDIT_CARD" for d in redactor.detect(text))

    def test_valid_luhn_card_is_still_detected(self, redactor):
        detected = redactor.detect("pay with 4111 1111 1111 1111 today")
        assert [d["type"] for d in detected] == ["CREDIT_CARD"]

    def test_overlapping_matches_collapse_to_leftmost_longest(self):
        # Two custom rules that both match the same digits: the longer span
        # should win and the shorter overlapping ones should be dropped, so a
        # value is never partly re-masked by a second rule.
        redactor = PIIRedactor(patterns={"LONG": r"\d{6}", "SHORT": r"\d{3}"})
        findings = redactor.detect("id 123456 end")
        assert len(findings) == 1
        assert findings[0]["type"] == "LONG"
        assert redactor.validate("id 123456 end") == "id [LONG] end"
