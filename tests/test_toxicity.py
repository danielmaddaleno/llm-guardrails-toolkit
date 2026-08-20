"""Unit tests for the keyword-based toxicity detector."""

import pytest

from guardrails.pipeline import GuardrailViolation
from guardrails.validators.toxicity import ToxicityDetector


@pytest.fixture
def detector():
    return ToxicityDetector()


class TestToxicityDetector:
    def test_safe_text_passes_unchanged(self, detector):
        text = "Please summarize the quarterly sales report."
        assert detector.validate(text) == text

    def test_hate_speech_is_blocked(self, detector):
        with pytest.raises(GuardrailViolation):
            detector.validate("That comment was blatantly racist.")

    def test_detect_reports_category_name(self, detector):
        flagged = detector.detect("This is a xenophobic remark.")
        assert flagged == ["hate_speech"]

    def test_matching_is_case_insensitive(self, detector):
        # Patterns compile with IGNORECASE, so casing should not matter.
        with pytest.raises(GuardrailViolation):
            detector.validate("SEXIST language is not allowed here.")

    def test_threshold_requires_multiple_categories(self):
        # With a threshold of 2, a single category match should pass.
        detector = ToxicityDetector(threshold=2)
        assert detector.detect("That was a sexist joke.") == ["hate_speech"]
        detector.validate("That was a sexist joke.")

    def test_threshold_blocks_when_two_categories_hit(self):
        detector = ToxicityDetector(threshold=2)
        # Message touches both the hate_speech and self_harm categories.
        text = "The homophobic troll also posted about self-harm."
        assert set(detector.detect(text)) == {"hate_speech", "self_harm"}
        with pytest.raises(GuardrailViolation):
            detector.validate(text)

    def test_threshold_below_one_is_rejected(self):
        # A threshold of 0 would make validate() block clean text, since
        # len([]) >= 0 is always true. The constructor should refuse it.
        with pytest.raises(ValueError):
            ToxicityDetector(threshold=0)
        with pytest.raises(ValueError):
            ToxicityDetector(threshold=-1)

    def test_custom_categories_override_defaults(self):
        detector = ToxicityDetector(categories={"spam": [r"\bbuy now\b"]})
        with pytest.raises(GuardrailViolation):
            detector.validate("buy now while stocks last")
        # A default-category term is no longer flagged once categories are replaced.
        assert detector.detect("that was racist") == []

    def test_invisible_characters_do_not_hide_a_keyword(self, detector):
        # A zero-width space wedged inside the word keeps it readable to a human
        # but would slip past a literal regex. Detection normalizes first, so the
        # slur is still flagged.
        assert detector.detect("that was raci​st") == ["hate_speech"]
        with pytest.raises(GuardrailViolation):
            detector.validate("that was raci​st")

    def test_fullwidth_lookalikes_do_not_hide_a_keyword(self, detector):
        # Fullwidth Latin letters render like ASCII but are distinct code points;
        # NFKC folds them back before matching so the category still trips.
        assert detector.detect("that was ｒａｃｉｓｔ") == ["hate_speech"]
