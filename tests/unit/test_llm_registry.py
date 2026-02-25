"""
Unit tests for the LLM registry.

Does NOT make real API calls — tests only the routing logic.
"""

import pytest

from ontograph.llm.registry import get_provider, list_providers


class TestRegistry:
    def test_known_aliases(self):
        known = list_providers()
        assert "claude" in known
        assert "openai" in known
        assert "gemini" in known

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("nonexistent_provider_xyz")

    def test_claude_alias(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        provider = get_provider("claude")
        assert provider.provider_name == "anthropic"
        assert "claude" in provider.model_id

    def test_anthropic_alias_same_as_claude(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        p1 = get_provider("claude")
        p2 = get_provider("anthropic")
        assert p1.provider_name == p2.provider_name

    def test_model_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        provider = get_provider("claude", model="claude-opus-4-6")
        assert provider.model_id == "claude-opus-4-6"

    def test_env_default_provider(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        provider = get_provider()
        assert provider.provider_name == "openai"

    def test_env_default_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        monkeypatch.setenv("DEFAULT_LLM_MODEL", "claude-opus-4-6")
        provider = get_provider("claude")
        assert provider.model_id == "claude-opus-4-6"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        provider = get_provider("CLAUDE")
        assert provider.provider_name == "anthropic"
