"""AWS Bedrock integration: wraps invoke_model with guardrails."""

from __future__ import annotations

import json
import logging
from typing import Any

from guardrails.pipeline import GuardrailsPipeline, ValidationResult

logger = logging.getLogger(__name__)

# Request-body keys invoke() builds itself and will not take from a caller.
# max_tokens and temperature are named arguments, so they cannot reach **kwargs.
RESERVED_BODY_KEYS = frozenset({"anthropic_version", "messages"})


class BedrockGuardedClient:
    """Thin wrapper around ``boto3`` Bedrock runtime that applies
    a :class:`GuardrailsPipeline` to both prompts and responses.

    Parameters
    ----------
    pipeline : GuardrailsPipeline
        Pre-configured pipeline instance.
    model_id : str
        Bedrock model identifier, e.g. ``"anthropic.claude-3-sonnet-20240229-v1:0"``.
    region_name : str
        AWS region.  Defaults to ``"us-east-1"``.
    boto3_session : Any | None
        Optional pre-existing ``boto3.Session``.
    """

    def __init__(
        self,
        pipeline: GuardrailsPipeline,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1",
        boto3_session: Any | None = None,
    ):
        self.pipeline = pipeline
        self.model_id = model_id

        session = boto3_session
        if session is None:
            # Only needed when we have to build the session ourselves. A caller
            # that injects its own session (or a stub) does not need boto3 at all.
            try:
                import boto3  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "boto3 is required for BedrockGuardedClient. " "Install it with: pip install boto3"
                ) from exc

            session = boto3.Session(region_name=region_name)

        self.client = session.client("bedrock-runtime")

    # ------------------------------------------------------------------
    def invoke(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send *prompt* through input guardrails, then Bedrock, then output guardrails.

        Returns
        -------
        dict with keys ``"text"``, ``"blocked"``, ``"stage"``, ``"input_validation"``,
        ``"output_validation"``, and ``"usage"`` (raw Bedrock usage metadata).

        Raises:
            ValueError: If *kwargs* carries a key this method owns.
        """
        clashing = RESERVED_BODY_KEYS.intersection(kwargs)
        if clashing:
            raise ValueError(
                f"{sorted(clashing)} cannot be passed to invoke(): the request body is built from "
                "the guarded prompt, and overriding it would send unvalidated text to the model."
            )

        # --- Input guardrails ---
        input_result: ValidationResult = self.pipeline.validate_input_full(prompt)
        if not input_result.is_safe:
            return {
                "text": None,
                "blocked": True,
                "stage": "input",
                "input_validation": input_result,
                "output_validation": None,
                "usage": None,
            }

        safe_prompt = input_result.processed_text

        # --- Call Bedrock ---
        # kwargs go first so no caller key can overwrite the guarded prompt.
        body = json.dumps(
            {
                **kwargs,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": safe_prompt}],
            }
        )

        logger.info("Invoking Bedrock model %s", self.model_id)
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        response_body = json.loads(response["body"].read())
        raw_text: str = response_body["content"][0]["text"]

        # --- Output guardrails ---
        output_result: ValidationResult = self.pipeline.validate_output_full(raw_text)

        return {
            "text": output_result.processed_text if output_result.is_safe else None,
            "blocked": not output_result.is_safe,
            "stage": "output" if not output_result.is_safe else None,
            "input_validation": input_result,
            "output_validation": output_result,
            "usage": response_body.get("usage"),
        }
