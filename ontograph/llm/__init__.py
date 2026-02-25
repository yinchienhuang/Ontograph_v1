"""
ontograph.llm — Multi-provider LLM abstraction layer.

All pipeline components call LLMs exclusively through this interface.
No free-text outputs: every call specifies a `response_model` (Pydantic class)
and receives a validated instance back.

Quick start:
    from ontograph.llm import get_provider, LLMRequest, LLMMessage

    provider = get_provider("claude")          # or "gpt-4o", "gemini"
    response = provider.complete(LLMRequest(
        messages=[LLMMessage(role="user", content="...")],
        response_model=MyOutputSchema,
    ))
    result: MyOutputSchema = response.parsed
"""

from ontograph.llm.base import (
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMSchemaError,
    TokenUsage,
)
from ontograph.llm.registry import get_provider, list_providers

__all__ = [
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMProvider",
    "TokenUsage",
    "LLMError",
    "LLMSchemaError",
    "get_provider",
    "list_providers",
]
