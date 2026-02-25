"""
llm/anthropic.py — Anthropic (Claude) provider adapter.

Structured output is achieved via the tool_use mechanism:
  - We define a single tool whose `input_schema` is the JSON Schema of
    `request.response_model`.
  - We set `tool_choice={"type": "tool", "name": TOOL_NAME}` to force the
    model to always call it.
  - The response's `tool_use` block input is the structured JSON we parse.
"""

from __future__ import annotations

import json

import anthropic

from ontograph.llm.base import (
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMSchemaError,
    TokenUsage,
)

_TOOL_NAME = "structured_output"


class AnthropicProvider:
    """Claude provider using tool_use for guaranteed structured JSON output."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)  # reads ANTHROPIC_API_KEY if None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        schema = request.response_model.model_json_schema()

        # Anthropic tool_use: force the model to fill exactly this schema
        tool = {
            "name": _TOOL_NAME,
            "description": (
                "Return structured data. You MUST call this tool with all required fields."
            ),
            "input_schema": schema,
        }

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Separate system message if present
        system: str | anthropic.NotGiven = anthropic.NOT_GIVEN
        if messages and messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system,
                messages=messages,  # type: ignore[arg-type]
                tools=[tool],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": _TOOL_NAME},
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        # Extract the tool_use block
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )
        if tool_block is None:
            raise LLMSchemaError("No tool_use block in Anthropic response")

        raw_json = json.dumps(tool_block.input)

        try:
            parsed = request.response_model.model_validate(tool_block.input)
        except Exception as exc:
            raise LLMSchemaError(
                f"Anthropic response failed validation: {exc}\nRaw: {raw_json}"
            ) from exc

        return LLMResponse(
            parsed=parsed,
            raw_json=raw_json,
            model_id=self._model,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )
