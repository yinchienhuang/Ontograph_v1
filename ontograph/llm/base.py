"""
llm/base.py — Provider protocol and shared request/response types.

Every LLM call in the pipeline goes through this interface.
All responses are structured JSON validated against a Pydantic model —
there are no free-text LLM outputs anywhere in the codebase.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class LLMMessage(BaseModel):
    role: str = Field(description="'system' | 'user' | 'assistant'")
    content: str


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class LLMRequest(BaseModel):
    """
    A single LLM call specification.

    `response_model` is a Pydantic class.  Each provider adapter calls
    `response_model.model_json_schema()` and converts it to its own
    native structured-output mechanism:

        Anthropic  → tools=[{input_schema: schema}] + tool_use parsing
        OpenAI     → response_format={"type":"json_schema","json_schema":…}
        Gemini     → generation_config.response_schema = schema
    """

    messages: list[LLMMessage]
    response_model: type[BaseModel] = Field(
        description="Pydantic class whose schema drives structured output"
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """
    The parsed and validated result of an LLM call.

    `parsed` is a validated instance of `LLMRequest.response_model`.
    `raw_json` is kept for debugging and audit logging.
    """

    parsed: Any = Field(description="Validated instance of request.response_model")
    raw_json: str
    model_id: str
    usage: TokenUsage

    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """
    Minimal interface every provider adapter must satisfy.

    Implementations live in anthropic.py / openai.py / gemini.py.
    Use registry.get_provider() to obtain an instance by name.
    """

    @property
    def provider_name(self) -> str:
        """Human-readable name, e.g. 'anthropic', 'openai', 'gemini'."""
        ...

    @property
    def model_id(self) -> str:
        """Exact model string sent to the API, e.g. 'claude-sonnet-4-6'."""
        ...

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Execute one structured LLM call.

        Must raise `LLMError` on API failures.
        Must raise `LLMSchemaError` if the response can't be validated
        against `request.response_model`.
        """
        ...


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    """Raised when the provider API returns an error."""


class LLMSchemaError(ValueError):
    """Raised when the LLM response fails Pydantic validation."""
