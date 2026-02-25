"""
llm/gemini.py — Google Gemini provider adapter.

Structured output is achieved via generation_config.response_schema.
The schema is derived from the Pydantic model's JSON Schema.

Docs: https://ai.google.dev/gemini-api/docs/structured-output
"""

from __future__ import annotations

import json

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from ontograph.llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMSchemaError,
    TokenUsage,
)


class GeminiProvider:
    """Gemini provider using response_schema for structured output."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: str | None = None,
    ) -> None:
        self._model_name = model
        if api_key:
            genai.configure(api_key=api_key)
        # else relies on GOOGLE_API_KEY env var picked up by the SDK

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_id(self) -> str:
        return self._model_name

    def complete(self, request: LLMRequest) -> LLMResponse:
        schema = request.response_model.model_json_schema()

        # Build prompt: Gemini uses a single content string or parts list.
        # We concatenate system + user messages; assistant turns are passed
        # as model role parts for multi-turn contexts.
        contents: list[dict[str, str]] = []
        for msg in request.messages:
            role = "model" if msg.role == "assistant" else msg.role
            # Gemini does not support a standalone "system" role in contents;
            # prepend system content to the first user message.
            if msg.role == "system":
                contents.append({"role": "user", "parts": msg.content})
            else:
                contents.append({"role": role, "parts": msg.content})

        model = genai.GenerativeModel(model_name=self._model_name)

        generation_config = GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )

        try:
            response = model.generate_content(
                contents=contents,
                generation_config=generation_config,
            )
        except Exception as exc:
            raise LLMError(f"Gemini API error: {exc}") from exc

        raw_json: str = response.text or ""

        if not raw_json:
            raise LLMSchemaError("Gemini returned empty content")

        try:
            data = json.loads(raw_json)
            parsed = request.response_model.model_validate(data)
        except Exception as exc:
            raise LLMSchemaError(
                f"Gemini response failed validation: {exc}\nRaw: {raw_json}"
            ) from exc

        # Gemini SDK exposes usage metadata differently across versions
        usage_meta = getattr(response, "usage_metadata", None)
        return LLMResponse(
            parsed=parsed,
            raw_json=raw_json,
            model_id=self._model_name,
            usage=TokenUsage(
                input_tokens=getattr(usage_meta, "prompt_token_count", 0),
                output_tokens=getattr(usage_meta, "candidates_token_count", 0),
            ),
        )
