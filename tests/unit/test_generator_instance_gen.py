"""Tests for ontograph/generator/instance_gen.py"""

from unittest.mock import MagicMock

import pytest

from ontograph.generator.instance_gen import (
    _has_ancestor,
    _concrete_components,
    _concrete_subsystems,
    _concrete_systems,
    generate_system,
)
from ontograph.generator.schema import (
    GeneratedAttribute,
    GeneratedSystem,
    GeneratedSystemBundle,
)
from ontograph.generator.taxonomy import AEROSPACE_TAXONOMY, PREDEFINED_DOMAINS
from ontograph.llm.base import LLMResponse, TokenUsage

NS = "http://example.org/test#"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_response(system: GeneratedSystem) -> LLMResponse:
    bundle = GeneratedSystemBundle(system=system)
    return LLMResponse(
        parsed=bundle,
        raw_json=bundle.model_dump_json(),
        model_id="test-model",
        usage=TokenUsage(input_tokens=100, output_tokens=200),
    )


def _sample_system() -> GeneratedSystem:
    return GeneratedSystem(
        local_name="NanoSatellite_Test",
        class_local="NanoSatellite",
        label="Test CubeSat",
        comment="A test nanosatellite.",
        attributes=[GeneratedAttribute(property_local="massKg", value="1.5")],
        subsystems=[],
    )


def _mock_provider(system: GeneratedSystem) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = _make_response(system)
    return provider


# ── Ancestry helpers ──────────────────────────────────────────────────────────

class TestHasAncestor:

    def test_solar_panel_descends_from_power_subsystem(self):
        assert _has_ancestor(AEROSPACE_TAXONOMY, "SolarPanel", "PowerSubsystem")

    def test_solar_panel_descends_from_subsystem(self):
        assert _has_ancestor(AEROSPACE_TAXONOMY, "SolarPanel", "Subsystem")

    def test_nano_satellite_not_subsystem(self):
        assert not _has_ancestor(AEROSPACE_TAXONOMY, "NanoSatellite", "Subsystem")

    def test_power_subsystem_descends_from_subsystem(self):
        assert _has_ancestor(AEROSPACE_TAXONOMY, "PowerSubsystem", "Subsystem")

    def test_unknown_class_returns_false(self):
        assert not _has_ancestor(AEROSPACE_TAXONOMY, "NonExistentClass", "Subsystem")


class TestConcreteHelpers:

    def test_concrete_components_nonempty(self):
        result = _concrete_components(AEROSPACE_TAXONOMY)
        assert len(result) > 0

    def test_concrete_components_excludes_abstract(self):
        result = set(_concrete_components(AEROSPACE_TAXONOMY))
        assert "Component" not in result
        assert "PowerComponent" not in result

    def test_concrete_components_domain_filter(self):
        cubesat_comps = set(_concrete_components(AEROSPACE_TAXONOMY, domain="cubesat"))
        # SolarPanel is cubesat; PassengerCabin is not
        assert "SolarPanel" in cubesat_comps
        assert "PassengerCabin" not in cubesat_comps

    def test_concrete_subsystems_nonempty(self):
        result = _concrete_subsystems(AEROSPACE_TAXONOMY)
        assert len(result) > 0
        assert "Subsystem" not in result
        assert "PowerSubsystem" in result

    def test_concrete_systems_nonempty(self):
        result = _concrete_systems(AEROSPACE_TAXONOMY)
        assert len(result) > 0
        assert "AerospaceSystem" not in result
        assert "NanoSatellite" in result


# ── generate_system ───────────────────────────────────────────────────────────

class TestGenerateSystem:

    def test_calls_provider_once(self):
        provider = _mock_provider(_sample_system())
        generate_system("cubesat", AEROSPACE_TAXONOMY, provider, NS)
        provider.complete.assert_called_once()

    def test_returns_correct_system(self):
        sample = _sample_system()
        provider = _mock_provider(sample)
        result = generate_system("cubesat", AEROSPACE_TAXONOMY, provider, NS)
        assert result.local_name == "NanoSatellite_Test"
        assert result.class_local == "NanoSatellite"

    def test_predefined_domain_prompt_contains_required_subsystems(self):
        provider = _mock_provider(_sample_system())
        generate_system("cubesat", AEROSPACE_TAXONOMY, provider, NS)
        request = provider.complete.call_args[0][0]
        system_msg = request.messages[0].content
        assert "PowerSubsystem" in system_msg
        assert "CommunicationSubsystem" in system_msg
        assert "AttitudeControlSubsystem" in system_msg

    def test_predefined_domain_prompt_contains_system_classes(self):
        provider = _mock_provider(_sample_system())
        generate_system("cubesat", AEROSPACE_TAXONOMY, provider, NS)
        request = provider.complete.call_args[0][0]
        system_msg = request.messages[0].content
        assert "NanoSatellite" in system_msg or "MicroSatellite" in system_msg

    def test_custom_domain_prompt_contains_full_vocabulary(self):
        provider = _mock_provider(_sample_system())
        generate_system("hypersonic_vehicle", AEROSPACE_TAXONOMY, provider, NS)
        request = provider.complete.call_args[0][0]
        system_msg = request.messages[0].content
        # Full vocabulary: classes from multiple domains present
        assert "NanoSatellite" in system_msg
        assert "HeavyLiftVehicle" in system_msg
        assert "eVTOL" in system_msg

    def test_request_uses_system_bundle_response_model(self):
        provider = _mock_provider(_sample_system())
        generate_system("rocket", AEROSPACE_TAXONOMY, provider, NS)
        request = provider.complete.call_args[0][0]
        assert request.response_model is GeneratedSystemBundle

    def test_request_temperature_matches_default(self):
        provider = _mock_provider(_sample_system())
        generate_system("uam", AEROSPACE_TAXONOMY, provider, NS, temperature=0.5)
        request = provider.complete.call_args[0][0]
        assert request.temperature == 0.5

    def test_instance_count_appears_in_prompt(self):
        provider = _mock_provider(_sample_system())
        generate_system("cubesat", AEROSPACE_TAXONOMY, provider, NS, instance_count=25)
        request = provider.complete.call_args[0][0]
        system_msg = request.messages[0].content
        assert "25" in system_msg

    def test_instance_count_appears_in_user_message(self):
        provider = _mock_provider(_sample_system())
        generate_system("rocket", AEROSPACE_TAXONOMY, provider, NS, instance_count=30)
        request = provider.complete.call_args[0][0]
        user_msg = request.messages[1].content
        assert "30" in user_msg

    def test_default_instance_count_is_reasonable(self):
        """Default instance_count should be >= 1 (smoke-test the default path)."""
        provider = _mock_provider(_sample_system())
        generate_system("lunar", AEROSPACE_TAXONOMY, provider, NS)
        provider.complete.assert_called_once()
