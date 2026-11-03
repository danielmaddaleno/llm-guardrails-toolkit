"""Tests for the model-based ClassifierValidator."""

import asyncio

import pytest

from guardrails import ClassifierValidator, GuardrailsPipeline
from guardrails.pipeline import GuardrailViolation


def const(score):
    """A scorer that ignores the text and returns a fixed score."""
    return lambda text: score


class TestConstruction:
    def test_rejects_non_callable_score_fn(self):
        with pytest.raises(TypeError):
            ClassifierValidator(score_fn=0.9)

    def test_rejects_unknown_severity(self):
        with pytest.raises(ValueError, match="severity"):
            ClassifierValidator(score_fn=const(0.0), severity="loud")

    def test_name_defaults_to_class_name(self):
        assert ClassifierValidator(score_fn=const(0.0)).name == "ClassifierValidator"

    def test_name_override(self):
        assert ClassifierValidator(score_fn=const(0.0), name="toxicity-model").name == "toxicity-model"


class TestSyncScoring:
    def test_below_threshold_passes_text_through(self):
        guard = ClassifierValidator(score_fn=const(0.3), threshold=0.5)
        assert guard.validate("hello") == "hello"

    def test_at_threshold_blocks(self):
        guard = ClassifierValidator(score_fn=const(0.5), threshold=0.5)
        with pytest.raises(GuardrailViolation) as exc:
            guard.validate("hello")
        assert exc.value.severity == "block"

    def test_above_threshold_blocks(self):
        guard = ClassifierValidator(score_fn=const(0.95), threshold=0.5)
        with pytest.raises(GuardrailViolation):
            guard.validate("hello")

    def test_scorer_sees_the_text(self):
        seen = []

        def record(text):
            seen.append(text)
            return 0.0

        ClassifierValidator(score_fn=record).validate("inspect me")
        assert seen == ["inspect me"]

    def test_message_reports_score_not_text(self):
        guard = ClassifierValidator(score_fn=const(0.9), threshold=0.5)
        with pytest.raises(GuardrailViolation) as exc:
            guard.validate("secret payload the user wrote")
        assert "secret payload" not in exc.value.message
        assert "0.900" in exc.value.message

    def test_warn_severity_carries_through(self):
        guard = ClassifierValidator(score_fn=const(0.9), severity="warn")
        with pytest.raises(GuardrailViolation) as exc:
            guard.validate("hello")
        assert exc.value.severity == "warn"

    def test_sync_path_rejects_async_scorer(self):
        made = []

        async def _coro():
            return 0.9

        def scorer(text):
            coro = _coro()
            made.append(coro)
            return coro

        guard = ClassifierValidator(score_fn=scorer)
        with pytest.raises(TypeError, match="async"):
            guard.validate("hello")
        for coro in made:
            coro.close()  # the guard rejected it unawaited; close it so the loop is clean


class TestAsyncScoring:
    def test_async_scorer_below_threshold_passes(self):
        async def scorer(text):
            await asyncio.sleep(0)
            return 0.1

        guard = ClassifierValidator(score_fn=scorer, threshold=0.5)
        assert asyncio.run(guard.validate_async("hello")) == "hello"

    def test_async_scorer_blocks(self):
        async def scorer(text):
            await asyncio.sleep(0)
            return 0.8

        guard = ClassifierValidator(score_fn=scorer, threshold=0.5)
        with pytest.raises(GuardrailViolation):
            asyncio.run(guard.validate_async("hello"))

    def test_sync_scorer_runs_on_async_path(self):
        guard = ClassifierValidator(score_fn=const(0.2), threshold=0.5)
        assert asyncio.run(guard.validate_async("hello")) == "hello"


class TestPipelineIntegration:
    def test_block_aborts_pipeline(self):
        pipe = GuardrailsPipeline(input_guards=[ClassifierValidator(score_fn=const(0.9))])
        with pytest.raises(GuardrailViolation):
            pipe.validate_input("hello")

    def test_warn_is_recorded_but_safe(self):
        guard = ClassifierValidator(score_fn=const(0.9), severity="warn", name="moderation")
        result = GuardrailsPipeline(output_guards=[guard]).validate_output_full("hello")
        assert result.is_safe
        assert len(result.violations) == 1
        assert result.violations[0].validator == "moderation"

    def test_async_pipeline_awaits_the_scorer(self):
        scored = []

        async def scorer(text):
            await asyncio.sleep(0)
            scored.append(text)
            return 0.1

        pipe = GuardrailsPipeline(input_guards=[ClassifierValidator(score_fn=scorer)])
        assert asyncio.run(pipe.validate_input_async("check this")) == "check this"
        assert scored == ["check this"]
