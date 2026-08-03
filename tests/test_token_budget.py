"""Unit tests for the token budget validator."""

import pytest

from guardrails.pipeline import GuardrailViolation
from guardrails.validators.token_budget import TokenBudget


class TestTokenBudgetConstructor:
    def test_rejects_negative_max_tokens(self):
        with pytest.raises(ValueError):
            TokenBudget(max_tokens=-1)

    def test_rejects_non_positive_chars_per_token(self):
        with pytest.raises(ValueError):
            TokenBudget(chars_per_token=0)
        with pytest.raises(ValueError):
            TokenBudget(chars_per_token=-4.0)

    def test_zero_max_tokens_is_allowed(self):
        # A budget of zero is a valid (if strict) limit: empty text still fits,
        # any real content does not.
        budget = TokenBudget(max_tokens=0)
        assert budget.validate("") == ""
        with pytest.raises(GuardrailViolation):
            budget.validate("a")


class TestTokenBudgetValidate:
    def test_text_within_budget_passes_through_unchanged(self):
        budget = TokenBudget(max_tokens=10, chars_per_token=4.0)
        text = "short enough"
        assert budget.validate(text) == text

    def test_text_over_budget_is_blocked(self):
        budget = TokenBudget(max_tokens=5, chars_per_token=4.0)
        with pytest.raises(GuardrailViolation):
            budget.validate("A" * 100)

    def test_exact_limit_passes_but_one_over_blocks(self):
        # 20 chars / 4 per token = 5 tokens, exactly the limit, so it passes.
        budget = TokenBudget(max_tokens=5, chars_per_token=4.0)
        assert budget.validate("A" * 20) == "A" * 20
        # 21 chars = 5.25 tokens, which rounds up to 6 and trips the guard.
        with pytest.raises(GuardrailViolation):
            budget.validate("A" * 21)


class TestTokenBudgetEstimate:
    def test_empty_text_is_zero_tokens(self):
        assert TokenBudget().estimate_tokens("") == 0

    def test_fractional_estimate_rounds_up(self):
        # A single character is well under one token's worth of characters but
        # must still count as one: rounding down would let the running total
        # undercount and slip past a hard budget.
        budget = TokenBudget(chars_per_token=4.0)
        assert budget.estimate_tokens("a") == 1
        assert budget.estimate_tokens("a" * 5) == 2
