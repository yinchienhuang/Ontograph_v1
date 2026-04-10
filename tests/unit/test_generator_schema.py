"""Tests for ontograph/generator/schema.py"""

from ontograph.generator.schema import (
    GeneratedAttribute,
    GeneratedComponent,
    GeneratedSubsystem,
    GeneratedSystem,
    GeneratedSystemBundle,
)


class TestGeneratedAttribute:

    def test_defaults_to_decimal_datatype(self):
        attr = GeneratedAttribute(property_local="massKg", value="5.0")
        assert attr.datatype == "decimal"

    def test_explicit_string_datatype(self):
        attr = GeneratedAttribute(
            property_local="orbitType", value="LEO", datatype="string"
        )
        assert attr.datatype == "string"

    def test_explicit_integer_datatype(self):
        attr = GeneratedAttribute(
            property_local="storageGB", value="64", datatype="integer"
        )
        assert attr.datatype == "integer"


class TestGeneratedSystem:

    def test_empty_subsystems_is_valid(self):
        sys = GeneratedSystem(
            local_name="TestSat_1",
            class_local="NanoSatellite",
            label="Test Satellite",
            comment="A test satellite for validation.",
        )
        assert sys.subsystems == []
        assert sys.attributes == []

    def test_system_with_attributes(self):
        sys = GeneratedSystem(
            local_name="TestSat_2",
            class_local="MicroSatellite",
            label="Test MicroSat",
            comment="A larger test satellite.",
            attributes=[
                GeneratedAttribute(property_local="massKg", value="12.0"),
                GeneratedAttribute(property_local="missionType", value="EO", datatype="string"),
            ],
        )
        assert len(sys.attributes) == 2
        assert sys.attributes[0].value == "12.0"


class TestGeneratedSystemBundle:

    def test_bundle_roundtrip_json(self):
        bundle = GeneratedSystemBundle(
            system=GeneratedSystem(
                local_name="TestSat_1",
                class_local="NanoSatellite",
                label="Test",
                comment="Test satellite",
                attributes=[GeneratedAttribute(property_local="massKg", value="1.5")],
                subsystems=[],
            )
        )
        dumped = bundle.model_dump()
        restored = GeneratedSystemBundle.model_validate(dumped)
        assert restored.system.local_name == "TestSat_1"
        assert restored.system.attributes[0].value == "1.5"
        assert restored.system.attributes[0].datatype == "decimal"

    def test_bundle_with_full_hierarchy(self):
        bundle = GeneratedSystemBundle(
            system=GeneratedSystem(
                local_name="CubeSat_Test_1",
                class_local="NanoSatellite",
                label="Test CubeSat",
                comment="Full hierarchy test.",
                subsystems=[
                    GeneratedSubsystem(
                        local_name="PowerSub_Test_1",
                        class_local="PowerSubsystem",
                        label="Power",
                        comment="Power subsystem.",
                        components=[
                            GeneratedComponent(
                                local_name="Solar_Test_1",
                                class_local="SolarPanel",
                                label="Solar Panel",
                                comment="Triple-junction GaAs panel.",
                                attributes=[
                                    GeneratedAttribute(
                                        property_local="powerGenerationW", value="8.5"
                                    )
                                ],
                            )
                        ],
                    )
                ],
            )
        )
        sys = bundle.system
        assert len(sys.subsystems) == 1
        assert sys.subsystems[0].local_name == "PowerSub_Test_1"
        assert len(sys.subsystems[0].components) == 1
        assert sys.subsystems[0].components[0].class_local == "SolarPanel"
