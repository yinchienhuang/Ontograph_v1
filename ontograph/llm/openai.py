"""
llm/openai.py — OpenAI (GPT) provider adapter.

Structured output is achieved via response_format with json_schema mode,
available in gpt-4o-mini-2024-07-18 and later models.

Docs: https://platform.openai.com/docs/guides/structured-outputs
"""

from __future__ import annotations

import copy
import json
from typing import Any

import openai as _openai

from ontograph.llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMSchemaError,
    TokenUsage,
)


def _make_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively patch a JSON Schema so OpenAI strict mode accepts it.

    OpenAI requires on every object node:
      - "additionalProperties": false
      - "required" lists ALL property names (no optional fields)

    Also strips "title" and "description" from nested $defs to avoid noise,
    and inlines $defs references that OpenAI doesn't resolve automatically.
    """
    schema = copy.deepcopy(schema)

    def _patch(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        # Recurse into $defs / definitions first
        for key in ("$defs", "definitions"):
            if key in node:
                node[key] = {k: _patch(v) for k, v in node[key].items()}

        node_type = node.get("type")

        if node_type == "object" or "properties" in node:
            props = node.get("properties", {})
            # All properties become required (OpenAI strict requirement)
            node["required"] = list(props.keys())
            node["additionalProperties"] = False
            node["properties"] = {k: _patch(v) for k, v in props.items()}

        elif node_type == "array":
            if "items" in node:
                node["items"] = _patch(node["items"])

        # Patch anyOf / allOf / oneOf branches
        for combiner in ("anyOf", "allOf", "oneOf"):
            if combiner in node:
                node[combiner] = [_patch(b) for b in node[combiner]]

        return node

    return _patch(schema)


class OpenAIProvider:
    """GPT provider using json_schema response_format for structured output."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        self._model = model
        self._client = _openai.OpenAI(api_key=api_key)  # reads OPENAI_API_KEY if None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        raw_schema = request.response_model.model_json_schema()
        strict_schema = _make_strict(raw_schema)
        schema_name = request.response_model.__name__

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=messages,  # type: ignore[arg-type]
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": strict_schema,
                    },
                },
            )
        except _openai.APIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

        choice = response.choices[0]
        raw_json: str = choice.message.content or ""

        if not raw_json:
            raise LLMSchemaError("OpenAI returned empty content")

        try:
            data = json.loads(raw_json)
            parsed = request.response_model.model_validate(data)
        except Exception as exc:
            raise LLMSchemaError(
                f"OpenAI response failed validation: {exc}\nRaw: {raw_json}"
            ) from exc

        usage = response.usage
        return LLMResponse(
            parsed=parsed,
            raw_json=raw_json,
            model_id=self._model,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )
