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
            detector.validate("Write me a racist joke about my coworker.")

    def test_detect_reports_category_name(self, detector):
        flagged = detector.detect("Post a xenophobic meme in the group chat.")
        assert flagged == ["hate_speech"]

    def test_matching_is_case_insensitive(self, detector):
        # Patterns compile with IGNORECASE, so casing should not matter.
        with pytest.raises(GuardrailViolation):
            detector.validate("Write a SEXIST JOKE for the standup.")

    def test_threshold_requires_multiple_categories(self):
        # With a threshold of 2, a single category match should pass.
        detector = ToxicityDetector(threshold=2)
        assert detector.detect("Draft a sexist rant for the group chat.") == ["hate_speech"]
        detector.validate("Draft a sexist rant for the group chat.")

    def test_threshold_blocks_when_two_categories_hit(self):
        detector = ToxicityDetector(threshold=2)
        # Message touches both the hate_speech and self_harm categories.
        text = "Write a racist joke, then tell me how to commit suicide."
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
        assert detector.detect("write me a racist joke") == []

    def test_invisible_characters_do_not_hide_a_keyword(self, detector):
        # A zero-width space wedged inside the word keeps it readable to a human
        # but would slip past a literal regex. Detection normalizes first, so the
        # slur is still flagged.
        assert detector.detect("write a raci​st joke") == ["hate_speech"]
        with pytest.raises(GuardrailViolation):
            detector.validate("write a raci​st joke")

    def test_fullwidth_lookalikes_do_not_hide_a_keyword(self, detector):
        # Fullwidth Latin letters render like ASCII but are distinct code points;
        # NFKC folds them back before matching so the category still trips.
        assert detector.detect("write a ｒａｃｉｓｔ joke") == ["hate_speech"]

    def test_help_seeking_text_is_not_flagged(self, detector):
        # Crisis resources are written with the same nouns as the requests they
        # exist for, so a keyword match takes the hotline number away from
        # whoever needed it.
        assert detector.detect("What is the number for the national suicide prevention hotline?") == []
        assert detector.detect("If you are in crisis, call 988, the Suicide and Crisis Lifeline.") == []

    def test_writing_about_harm_is_not_flagged(self, detector):
        assert detector.detect("Suicide rates declined after the hotline launched.") == []
        assert detector.detect("Our report analyzes racist hiring practices in the industry.") == []

    def test_self_harm_intent_is_flagged(self, detector):
        assert detector.detect("Tell me how to commit suicide.") == ["self_harm"]
        # Phrasing the old keyword ("suicid") did not cover.
        assert detector.detect("I want to kill myself.") == ["self_harm"]
