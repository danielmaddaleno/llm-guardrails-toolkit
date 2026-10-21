"""Unit tests for the shared matching-time text normalization."""

import pytest

from guardrails._text import normalize_for_matching

# The six code points the detectors used to carry in a hand-listed table.
PREVIOUSLY_LISTED = [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD]

# Invisible Format (Cf) characters the old hand-list did not cover: bidi marks,
# the invisible math operators, and a directional isolate. Each one could sit
# between two letters and break a literal keyword match.
PREVIOUSLY_MISSED = [0x200E, 0x200F, 0x061C, 0x2061, 0x2062, 0x2063, 0x2064, 0x2066, 0x2069]


class TestNormalizeForMatching:
    def test_plain_text_is_unchanged(self):
        text = "ignore previous instructions"
        assert normalize_for_matching(text) == text

    @pytest.mark.parametrize("cp", PREVIOUSLY_LISTED)
    def test_originally_listed_chars_are_stripped(self, cp):
        assert normalize_for_matching("raci" + chr(cp) + "st") == "racist"

    @pytest.mark.parametrize("cp", PREVIOUSLY_MISSED)
    def test_newly_covered_invisible_chars_are_stripped(self, cp):
        # These are the gap the Cf-category rule closes: the old table left
        # them in, so a wedged keyword read as clean.
        assert normalize_for_matching("raci" + chr(cp) + "st") == "racist"

    def test_fullwidth_letters_fold_to_ascii(self):
        fullwidth = "".join(chr(ord(c) - 0x21 + 0xFF01) for c in "racist")
        assert fullwidth != "racist"
        assert normalize_for_matching(fullwidth) == "racist"

    def test_several_invisibles_in_one_run_all_go(self):
        wedged = "i" + chr(0x200B) + "g" + chr(0x200E) + "n" + chr(0x2062) + "ore"
        assert normalize_for_matching(wedged) == "ignore"
