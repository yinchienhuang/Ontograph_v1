"""
llm/registry.py — Provider registry.

Maps short provider names + optional model overrides to LLMProvider instances.

Usage:
    provider = get_provider("claude")
    provider = get_provider("gpt-4o")
    provider = get_provider("gemini", model="gemini-1.5-flash")
    provider = get_provider("claude", model="claude-opus-4-6")
"""

from __future__ import annotations

import os

from ontograph.llm.base import LLMProvider

# ---------------------------------------------------------------------------
# Default model IDs per provider
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, str] = {
    "claude":    "claude-sonnet-4-6",
    "anthropic": "claude-sonnet-4-6",
    "gpt-4o":    "gpt-4o",
    "openai":    "gpt-4o",
    "gemini":    "gemini-1.5-pro",
}

# Lazy import to avoid failing at import time when a package isn't installed
def _make_anthropic(model: str) -> LLMProvider:
    from ontograph.llm.anthropic import AnthropicProvider
    return AnthropicProvider(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))


def _make_openai(model: str) -> LLMProvider:
    from ontograph.llm.openai import OpenAIProvider
    return OpenAIProvider(model=model, api_key=os.getenv("OPENAI_API_KEY"))


def _make_gemini(model: str) -> LLMProvider:
    from ontograph.llm.gemini import GeminiProvider
    return GeminiProvider(model=model, api_key=os.getenv("GOOGLE_API_KEY"))


_FACTORIES: dict[str, object] = {
    "claude":    _make_anthropic,
    "anthropic": _make_anthropic,
    "gpt-4o":    _make_openai,
    "openai":    _make_openai,
    "gemini":    _make_gemini,
}


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    """
    Return an LLMProvider instance for the requested provider.

    Args:
        name:  Provider alias. If None, reads DEFAULT_LLM_PROVIDER from env
               (defaults to 'claude').
        model: Model override. If None, uses the provider's default model,
               unless DEFAULT_LLM_MODEL env var is set.

    Raises:
        ValueError: if the provider name is unknown.
    """
    resolved_name = (name or os.getenv("DEFAULT_LLM_PROVIDER", "claude")).lower()

    if resolved_name not in _FACTORIES:
        known = ", ".join(sorted(_DEFAULTS))
        raise ValueError(
            f"Unknown LLM provider '{resolved_name}'. Known: {known}"
        )

    resolved_model = (
        model
        or os.getenv("DEFAULT_LLM_MODEL")
        or _DEFAULTS[resolved_name]
    )

    factory = _FACTORIES[resolved_name]
    return factory(resolved_model)  # type: ignore[call-arg,return-value]


def list_providers() -> list[str]:
    """Return the list of known provider alias strings."""
    return sorted(_DEFAULTS)
