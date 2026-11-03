"""Validator that defers the verdict to a model-based classifier."""

from __future__ import annotations

import inspect
from typing import Awaitable, Callable, Optional, Union, cast

from guardrails.pipeline import BaseValidator, GuardrailViolation

_SEVERITIES = ("block", "warn")

# A scorer returns how harmful the text is. Returning an awaitable lets a
# hosted endpoint be called from the async pipeline; a plain float covers a
# local model that scores in-process.
ScoreFn = Callable[[str], Union[float, Awaitable[float]]]


class ClassifierValidator(BaseValidator):
    """Turn a trained classifier into a guard.

    The built-in detectors are regex, so they catch known phrasing and miss
    paraphrased or obfuscated content. The README recommends pairing them with
    a real classifier (a toxicity model, a hosted moderation endpoint) and the
    pipeline grew an async path for exactly that call. This validator is the
    seam between the two: give it a function that scores a string and a
    threshold, and it blocks (or warns) once the score reaches the line.

    The scorer may be synchronous or return an awaitable. An async scorer is
    awaited on the async pipeline, so a network round-trip to a moderation
    service yields the event loop rather than holding it. Calling the sync
    :meth:`validate` with an async scorer raises, because there is no loop to
    await on; route it through ``validate_*_async`` instead.

    Parameters
    ----------
    score_fn:
        Callable mapping text to a score. Higher means more likely harmful.
    threshold:
        Score at or above which the guard fires. Scores are compared as
        floats, so a 0-1 probability and a raw logit both work as long as the
        threshold is on the same scale.
    severity:
        ``"block"`` aborts the pipeline, ``"warn"`` records the hit and lets
        the text through. Warn is the safer default for an output guard, where
        a false positive would drop the whole response.
    name:
        Overrides the audit name, so several classifiers can sit in one
        pipeline and still be told apart in ``ValidationResult`` and logs.
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        threshold: float = 0.5,
        severity: str = "block",
        name: Optional[str] = None,
    ):
        if not callable(score_fn):
            raise TypeError("score_fn must be callable")
        if severity not in _SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(_SEVERITIES)}, got {severity!r}")
        self.score_fn = score_fn
        self.threshold = float(threshold)
        self.severity = severity
        self._name = name

    @property
    def name(self) -> str:
        return self._name or type(self).__name__

    def validate(self, text: str) -> str:
        result = self.score_fn(text)
        if inspect.isawaitable(result):
            raise TypeError(
                f"{self.name} has an async score_fn; run it through the async pipeline "
                "(validate_input_async / validate_output_async) so the score can be awaited."
            )
        self._enforce(cast(float, result))
        return text

    async def validate_async(self, text: str) -> str:
        result = self.score_fn(text)
        if inspect.isawaitable(result):
            score = await result
        else:
            score = cast(float, result)
        self._enforce(score)
        return text

    def _enforce(self, score: float) -> None:
        if score >= self.threshold:
            # Report the score, never the text, so the guard is not a second
            # place the flagged content gets copied.
            raise GuardrailViolation(
                validator=self.name,
                message=f"classifier score {score:.3f} is at or above the threshold of {self.threshold:.3f}",
                severity=self.severity,
            )
