# -*- coding: utf-8 -*-
# Author: Daniel Maddaleno
"""Pipeline — core implementation."""
"""Core guardrails pipeline orchestrator."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """Base class for all input/output validators."""

    @abstractmethod
    def validate(self, text: str) -> str:
        """Validate and optionally transform text.

        Args:
            text: The text to validate.

        Returns:
            Validated (possibly transformed) text.

        Raises:
            GuardrailViolation: If the text fails validation.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class GuardrailViolation(Exception):
    """Raised when a guardrail is triggered."""

    def __init__(self, validator: str, message: str, severity: str = "block"):
        self.validator = validator
        self.message = message
        self.severity = severity
        super().__init__(f"[{validator}] {message}")


@dataclass
class ValidationResult:
    """Result of a validation pipeline run."""

    original_text: str
    processed_text: str
    violations: list[GuardrailViolation] = field(default_factory=list)
    validators_applied: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return not any(v.severity == "block" for v in self.violations)


class GuardrailsPipeline:
    """Orchestrates input and output guardrails."""

    def __init__(
        self,
        input_guards: list[BaseValidator] | None = None,
        output_guards: list[BaseValidator] | None = None,
    ):
        self.input_guards = input_guards or []
        self.output_guards = output_guards or []

    def validate_input(self, text: str) -> str:
        """Run all input guardrails and return safe text."""
        return self._run_pipeline(text, self.input_guards, "input")

    def validate_output(self, text: str) -> str:
        """Run all output guardrails and return safe text."""
        return self._run_pipeline(text, self.output_guards, "output")

    def validate_input_full(self, text: str) -> ValidationResult:
        """Run input guardrails and return detailed result."""
        return self._run_pipeline_full(text, self.input_guards, "input")

    def _run_pipeline(self, text: str, validators: list[BaseValidator], stage: str) -> str:
        """Run validators sequentially, return processed text or raise."""
        current = text
        for validator in validators:
            try:
                current = validator.validate(current)
                logger.debug("[%s] %s passed", stage, validator.name)
            except GuardrailViolation:
                logger.warning("[%s] %s triggered", stage, validator.name)
                raise
        return current

    def _run_pipeline_full(
        self, text: str, validators: list[BaseValidator], stage: str
    ) -> ValidationResult:
        """Run all validators and collect results without short-circuiting."""
        result = ValidationResult(original_text=text, processed_text=text)

        for validator in validators:
            try:
                result.processed_text = validator.validate(result.processed_text)
                result.validators_applied.append(validator.name)
            except GuardrailViolation as e:
                result.violations.append(e)
                result.validators_applied.append(validator.name)

        return result
