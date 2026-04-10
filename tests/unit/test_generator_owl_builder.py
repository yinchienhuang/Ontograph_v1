"""Tests for ontograph/generator/owl_builder.py"""

import tempfile
from pathlib import Path

import pytest
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from ontograph.generator.owl_builder import build_owl_graph, serialize_owl
from ontograph.generator.schema import (
    GeneratedAttribute,
    GeneratedComponent,
    GeneratedSubsystem,
    GeneratedSystem,
)
from ontograph.generator.taxonomy import AEROSPACE_TAXONOMY

NS_STR = "http://example.org/test-aerospace#"
NS = Namespace(NS_STR)


def _make_simple_system() -> GeneratedSystem:
    return GeneratedSystem(
        local_name="TestSat_Alpha_1",
        class_local="NanoSatellite",
        label="Test Satellite Alpha",
        comment="A test CubeSat for unit testing.",
        attributes=[
            GeneratedAttribute(property_local="massKg",      value="2.5",              datatype="decimal"),
            GeneratedAttribute(property_local="missionType", value="Earth Observation", datatype="string"),
        ],
        subsystems=[
            GeneratedSubsystem(
                local_name="PowerSub_Alpha_1",
                class_local="PowerSubsystem",
                label="Power Subsystem Alpha",
                comment="Power subsystem for Test Alpha.",
                attributes=[
                    GeneratedAttribute(property_local="powerW", value="10.0", datatype="decimal"),
                ],
                components=[
                    GeneratedComponent(
                        local_name="Solar_Alpha_1",
                        class_local="SolarPanel",
                        label="Solar Panel Alpha",
                        comment="Triple-junction 3U solar panel.",
                        attributes=[
                            GeneratedAttribute(property_local="powerGenerationW", value="8.5", datatype="decimal"),
                        ],
                    ),
                ],
            ),
        ],
    )


@pytest.fixture(scope="module")
def graph() -> Graph:
    return build_owl_graph([_make_simple_system()], AEROSPACE_TAXONOMY, NS_STR)


class TestTBoxDeclarations:

    def test_tbox_has_all_taxonomy_classes(self, graph):
        for cls in AEROSPACE_TAXONOMY.classes:
            assert (NS[cls.local], RDF.type, OWL.Class) in graph, (
                f"Missing owl:Class triple for '{cls.local}'"
            )

    def test_tbox_subclass_axioms(self, graph):
        for cls in AEROSPACE_TAXONOMY.classes:
            if cls.parent:
                assert (NS[cls.local], RDFS.subClassOf, NS[cls.parent]) in graph, (
                    f"Missing rdfs:subClassOf for '{cls.local}' → '{cls.parent}'"
                )

    def test_tbox_data_properties(self, graph):
        for dp in AEROSPACE_TAXONOMY.data_properties:
            assert (NS[dp.local], RDF.type, OWL.DatatypeProperty) in graph, (
                f"Missing owl:DatatypeProperty for '{dp.local}'"
            )

    def test_tbox_object_properties(self, graph):
        for op in AEROSPACE_TAXONOMY.object_properties:
            assert (NS[op.local], RDF.type, OWL.ObjectProperty) in graph, (
                f"Missing owl:ObjectProperty for '{op.local}'"
            )


class TestABoxIndividuals:

    def test_system_individual_typed(self, graph):
        sys_uri = NS["TestSat_Alpha_1"]
        assert (sys_uri, RDF.type, OWL.NamedIndividual) in graph
        assert (sys_uri, RDF.type, NS["NanoSatellite"]) in graph

    def test_subsystem_individual_typed(self, graph):
        sub_uri = NS["PowerSub_Alpha_1"]
        assert (sub_uri, RDF.type, OWL.NamedIndividual) in graph
        assert (sub_uri, RDF.type, NS["PowerSubsystem"]) in graph

    def test_component_individual_typed(self, graph):
        comp_uri = NS["Solar_Alpha_1"]
        assert (comp_uri, RDF.type, OWL.NamedIndividual) in graph
        assert (comp_uri, RDF.type, NS["SolarPanel"]) in graph

    def test_subsystem_linked_via_has_subsystem(self, graph):
        sys_uri = NS["TestSat_Alpha_1"]
        sub_uri = NS["PowerSub_Alpha_1"]
        assert (sys_uri, NS["hasSubsystem"], sub_uri) in graph

    def test_component_linked_via_has_component(self, graph):
        sub_uri  = NS["PowerSub_Alpha_1"]
        comp_uri = NS["Solar_Alpha_1"]
        assert (sub_uri, NS["hasComponent"], comp_uri) in graph

    def test_attribute_decimal_datatype(self, graph):
        sys_uri = NS["TestSat_Alpha_1"]
        mass_triples = list(graph.triples((sys_uri, NS["massKg"], None)))
        assert len(mass_triples) == 1
        _, _, obj = mass_triples[0]
        assert hasattr(obj, "datatype") and obj.datatype == XSD.decimal

    def test_attribute_string_datatype(self, graph):
        sys_uri = NS["TestSat_Alpha_1"]
        mission_triples = list(graph.triples((sys_uri, NS["missionType"], None)))
        assert len(mission_triples) == 1
        _, _, obj = mission_triples[0]
        assert hasattr(obj, "datatype") and obj.datatype == XSD.string

    def test_attribute_on_component(self, graph):
        comp_uri = NS["Solar_Alpha_1"]
        pw_triples = list(graph.triples((comp_uri, NS["powerGenerationW"], None)))
        assert len(pw_triples) == 1


class TestSerialization:

    def test_serialize_owl_creates_parseable_file(self):
        g = build_owl_graph([_make_simple_system()], AEROSPACE_TAXONOMY, NS_STR)
        with tempfile.NamedTemporaryFile(suffix=".owl", delete=False) as f:
            out_path = Path(f.name)
        try:
            serialize_owl(g, out_path)
            assert out_path.exists()
            g2 = Graph()
            g2.parse(str(out_path), format="xml")
            assert len(g2) > 0
        finally:
            out_path.unlink(missing_ok=True)

    def test_multiple_systems_all_present(self):
        sys1 = _make_simple_system()
        sys2 = GeneratedSystem(
            local_name="Rocket_Test_2",
            class_local="SmallLaunchVehicle",
            label="Test Rocket 2",
            comment="Second system.",
            attributes=[GeneratedAttribute(property_local="massKg", value="15000.0")],
        )
        g = build_owl_graph([sys1, sys2], AEROSPACE_TAXONOMY, NS_STR)
        assert (NS["TestSat_Alpha_1"], RDF.type, OWL.NamedIndividual) in g
        assert (NS["Rocket_Test_2"],   RDF.type, OWL.NamedIndividual) in g
