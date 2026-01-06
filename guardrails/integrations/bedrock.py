# -*- coding: utf-8 -*-
# Author: Daniel Maddaleno
"""Bedrock — core implementation."""
"""AWS Bedrock integration – wraps invoke_model with guardrails."""

from __future__ import annotations

import json
import logging
from typing import Any

from guardrails.pipeline import GuardrailsPipeline, ValidationResult

logger = logging.getLogger(__name__)


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

        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for BedrockGuardedClient. "
                "Install it with: pip install boto3"
            ) from exc

        session = boto3_session or boto3.Session(region_name=region_name)
        self.client = session.client("bedrock-runtime")

    # ------------------------------------------------------------------
    def invoke(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send *prompt* through input guardrails → Bedrock → output guardrails.

        Returns
        -------
        dict with keys ``"text"``, ``"input_validation"``, ``"output_validation"``,
        and ``"usage"`` (raw Bedrock usage metadata).
        """
        # --- Input guardrails ---
        input_result: ValidationResult = self.pipeline.run(prompt)
        if input_result.blocked:
            return {
                "text": None,
                "blocked": True,
                "stage": "input",
                "input_validation": input_result.to_dict(),
                "output_validation": None,
                "usage": None,
            }

        safe_prompt = input_result.sanitised_text

        # --- Call Bedrock ---
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": safe_prompt}],
                **kwargs,
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
        output_result: ValidationResult = self.pipeline.run(raw_text)

        return {
            "text": output_result.sanitised_text if not output_result.blocked else None,
            "blocked": output_result.blocked,
            "stage": "output" if output_result.blocked else None,
            "input_validation": input_result.to_dict(),
            "output_validation": output_result.to_dict(),
            "usage": response_body.get("usage"),
        }
