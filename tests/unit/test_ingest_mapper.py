"""
tests/unit/test_ingest_mapper.py — Tests for ontograph/ingest/mapper.py

All LLM calls are replaced with a MockProvider that returns pre-built
EntityMappingResponse objects.
"""

from __future__ import annotations

import hashlib
import warnings

import pytest

from ontograph.ingest.mapper import (
    DEFAULT_NAMESPACE,
    EntityMappingResponse,
    ProposedTriple,
    _build_mapping_messages,
    _convert_mapping,
    _entry_id,
    _expand_datatype,
    _iri,
    map_to_delta,
)
from ontograph.llm.base import LLMResponse, TokenUsage
from ontograph.models.document import SourceLocator
from ontograph.models.extraction import (
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionBundle,
)
from ontograph.models.ontology import ChangeSource

_NS = DEFAULT_NAMESPACE
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_XSD = "http://www.w3.org/2001/XMLSchema#"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_locator() -> SourceLocator:
    return SourceLocator(
        source_id="sha000",
        source_path="/fake/doc.md",
        source_format="md",
        line_start=1,
        line_end=10,
    )


def _make_entity(
    text_span: str = "ThrusterModule_A",
    entity_type: str = "Component",
    attributes: list[ExtractedAttribute] | None = None,
    relationships: list[ExtractedRelationship] | None = None,
) -> ExtractedEntity:
    return ExtractedEntity(
        id="entity001",
        text_span=text_span,
        chunk_id="chunk001",
        source_locator=_make_locator(),
        entity_type=entity_type,
        attributes=attributes or [],
        relationships=relationships or [],
        extraction_method="llm",
        confidence=0.95,
        section_context="Propulsion > Thruster",
    )


def _make_relationship(predicate: str, target: str) -> ExtractedRelationship:
    return ExtractedRelationship(
        predicate=predicate,
        target=target,
        raw_text=f"uses {target}",
        chunk_id="chunk001",
        source_locator=_make_locator(),
    )


def _make_bundle(entities: list[ExtractedEntity]) -> ExtractionBundle:
    sha = hashlib.sha256("fake".encode()).hexdigest()
    return ExtractionBundle(
        id="bundle001",
        document_artifact_id="artifact001",
        entities=entities,
        extractor_version="0.1.0/mock-model",
        created_at="2025-01-01T00:00:00+00:00",
    )


def _make_provider(*responses: EntityMappingResponse):
    it = iter(responses)

    class MockProvider:
        provider_name = "mock"
        model_id = "mock-model"

        def complete(self, request):
            return LLMResponse(
                parsed=next(it),
                raw_json="{}",
                model_id="mock-model",
                usage=TokenUsage(input_tokens=0, output_tokens=0),
            )

    return MockProvider()


def _thruster_mapping() -> EntityMappingResponse:
    return EntityMappingResponse(
        subject_local_name="ThrusterModule_A",
        rdf_type="PropulsionSubsystem",
        triples=[
            ProposedTriple(
                predicate="hasDryMass",
                object="12.4",
                datatype="xsd:float",
                rationale="Dry mass from PDR",
                confidence=0.95,
            ),
            ProposedTriple(
                predicate="hasVacuumThrust",
                object="220",
                datatype="xsd:float",
                rationale="Vacuum thrust from PDR",
                confidence=0.95,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# _expand_datatype
# ---------------------------------------------------------------------------

class TestExpandDatatype:
    def test_xsd_float(self):
        assert _expand_datatype("xsd:float") == _XSD + "float"

    def test_xsd_integer(self):
        assert _expand_datatype("xsd:integer") == _XSD + "integer"

    def test_xsd_string(self):
        assert _expand_datatype("xsd:string") == _XSD + "string"

    def test_full_iri_passthrough(self):
        full = _XSD + "float"
        assert _expand_datatype(full) == full

    def test_none_returns_none(self):
        assert _expand_datatype(None) is None

    def test_non_xsd_non_uri_returns_none(self):
        assert _expand_datatype("float") is None


# ---------------------------------------------------------------------------
# _iri
# ---------------------------------------------------------------------------

class TestIri:
    def test_full_iri_passthrough(self):
        full = "http://example.org/ns#Foo"
        assert _iri(full, _NS) == full

    def test_rdf_type_special_case(self):
        assert _iri("rdf:type", _NS) == _RDF_TYPE

    def test_xsd_prefix_expanded(self):
        assert _iri("xsd:float", _NS) == _XSD + "float"

    def test_bare_name_gets_namespace(self):
        assert _iri("ThrusterModule_A", _NS) == _NS + "ThrusterModule_A"

    def test_spaces_become_underscores(self):
        result = _iri("thruster module", _NS)
        assert "thruster_module" in result or "thruster" in result

    def test_special_chars_stripped(self):
        result = _iri("Foo-Bar!", _NS)
        assert "!" not in result
        assert "-" not in result


# ---------------------------------------------------------------------------
# _entry_id
# ---------------------------------------------------------------------------

class TestEntryId:
    def test_returns_16_hex(self):
        eid = _entry_id("s", "p", "o")
        assert len(eid) == 16

    def test_deterministic(self):
        assert _entry_id("s", "p", "o") == _entry_id("s", "p", "o")

    def test_different_triple_differs(self):
        assert _entry_id("s", "p", "o1") != _entry_id("s", "p", "o2")


# ---------------------------------------------------------------------------
# _build_mapping_messages
# ---------------------------------------------------------------------------

class TestBuildMappingMessages:
    def test_two_messages(self):
        entity = _make_entity()
        msgs = _build_mapping_messages(entity, _NS)
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_user_message_contains_namespace(self):
        entity = _make_entity()
        msgs = _build_mapping_messages(entity, _NS)
        assert _NS in msgs[1].content

    def test_user_message_contains_entity_span(self):
        entity = _make_entity(text_span="ThrusterModule_A")
        msgs = _build_mapping_messages(entity, _NS)
        assert "ThrusterModule_A" in msgs[1].content

    def test_user_message_contains_attribute(self):
        attr = ExtractedAttribute(
            name="dry_mass", raw_text="12.4 kg", value="12.4", unit="kg",
            chunk_id="c1", source_locator=_make_locator(),
        )
        entity = _make_entity(attributes=[attr])
        msgs = _build_mapping_messages(entity, _NS)
        assert "dry_mass" in msgs[1].content
        assert "12.4 kg" in msgs[1].content

    def test_user_message_contains_relationship(self):
        rel = _make_relationship("usesBusProtocol", "I2C")
        entity = _make_entity(relationships=[rel])
        msgs = _build_mapping_messages(entity, _NS)
        assert "usesBusProtocol" in msgs[1].content
        assert "I2C" in msgs[1].content

    def test_user_message_no_relationships_shows_none(self):
        entity = _make_entity()
        msgs = _build_mapping_messages(entity, _NS)
        assert "(none)" in msgs[1].content


# ---------------------------------------------------------------------------
# _convert_mapping
# ---------------------------------------------------------------------------

class TestConvertMapping:
    def test_rdf_type_entry_always_present(self):
        entity = _make_entity()
        mapping = _thruster_mapping()
        entries = _convert_mapping(mapping, entity, _NS)
        predicates = [e.triple.predicate for e in entries]
        assert _RDF_TYPE in predicates

    def test_rdf_type_object_is_class_iri(self):
        entity = _make_entity()
        mapping = _thruster_mapping()
        entries = _convert_mapping(mapping, entity, _NS)
        type_entry = next(e for e in entries if e.triple.predicate == _RDF_TYPE)
        assert type_entry.triple.object == _NS + "PropulsionSubsystem"

    def test_subject_iri_has_namespace(self):
        entity = _make_entity()
        mapping = _thruster_mapping()
        entries = _convert_mapping(mapping, entity, _NS)
        assert all(e.triple.subject.startswith(_NS) for e in entries)

    def test_attribute_triple_has_datatype(self):
        entity = _make_entity()
        mapping = _thruster_mapping()
        entries = _convert_mapping(mapping, entity, _NS)
        attr_entries = [e for e in entries if e.triple.predicate != _RDF_TYPE]
        assert all(e.triple.datatype == _XSD + "float" for e in attr_entries)

    def test_all_entries_status_proposed(self):
        entity = _make_entity()
        entries = _convert_mapping(_thruster_mapping(), entity, _NS)
        assert all(e.status == "proposed" for e in entries)

    def test_all_entries_change_source_pipeline(self):
        entity = _make_entity()
        entries = _convert_mapping(_thruster_mapping(), entity, _NS)
        assert all(e.change_source == ChangeSource.PIPELINE for e in entries)

    def test_source_entity_id_set(self):
        entity = _make_entity()
        entries = _convert_mapping(_thruster_mapping(), entity, _NS)
        assert all(e.source_entity_id == entity.id for e in entries)

    def test_source_chunk_id_set(self):
        entity = _make_entity()
        entries = _convert_mapping(_thruster_mapping(), entity, _NS)
        assert all(e.source_chunk_id == entity.chunk_id for e in entries)

    def test_entry_count_rdf_type_plus_attributes(self):
        entity = _make_entity()
        mapping = _thruster_mapping()  # 2 attribute triples + 1 rdf:type
        entries = _convert_mapping(mapping, entity, _NS)
        assert len(entries) == 3

    def test_iri_object_not_given_datatype(self):
        """Objects without a datatype that look like identifiers become IRIs."""
        entity = _make_entity()
        mapping = EntityMappingResponse(
            subject_local_name="FeedSystem_A",
            rdf_type="FluidSubsystem",
            triples=[
                ProposedTriple(
                    predicate="hasPropellant",
                    object="Propellant_MON3_MMH",   # looks like a local name
                    datatype=None,
                    rationale="Propellant type",
                    confidence=0.9,
                )
            ],
        )
        entries = _convert_mapping(mapping, entity, _NS)
        attr_entry = next(e for e in entries if e.triple.predicate != _RDF_TYPE)
        assert attr_entry.triple.object.startswith(_NS)
        assert attr_entry.triple.datatype is None

    def test_object_property_triple_from_relationship(self):
        """A null-datatype triple whose object is a bare word becomes an IRI triple."""
        entity = _make_entity(
            text_span="OBC1",
            relationships=[_make_relationship("usesBusProtocol", "I2C")],
        )
        mapping = EntityMappingResponse(
            subject_local_name="OBC1",
            rdf_type="OnBoardComputer",
            triples=[
                ProposedTriple(
                    predicate="usesBusProtocol",
                    object="I2C",
                    datatype=None,
                    rationale="OBC1 communicates via I2C",
                    confidence=0.9,
                )
            ],
        )
        entries = _convert_mapping(mapping, entity, _NS)
        rel_entry = next(e for e in entries if "usesBusProtocol" in e.triple.predicate)
        assert rel_entry.triple.predicate == _NS + "usesBusProtocol"
        assert rel_entry.triple.object == _NS + "I2C"
        assert rel_entry.triple.datatype is None

    def test_multiple_object_property_triples(self):
        """Two bus protocol relationships produce two separate IRI triples."""
        entity = _make_entity(
            text_span="OBC1",
            relationships=[
                _make_relationship("usesBusProtocol", "I2C"),
                _make_relationship("usesBusProtocol", "SPI"),
            ],
        )
        mapping = EntityMappingResponse(
            subject_local_name="OBC1",
            rdf_type="OnBoardComputer",
            triples=[
                ProposedTriple(predicate="usesBusProtocol", object="I2C",
                               datatype=None, rationale="I2C bus", confidence=0.9),
                ProposedTriple(predicate="usesBusProtocol", object="SPI",
                               datatype=None, rationale="SPI bus", confidence=0.9),
            ],
        )
        entries = _convert_mapping(mapping, entity, _NS)
        rel_entries = [e for e in entries if "usesBusProtocol" in e.triple.predicate]
        objects = {e.triple.object for e in rel_entries}
        assert _NS + "I2C" in objects
        assert _NS + "SPI" in objects
        assert all(e.triple.datatype is None for e in rel_entries)


# ---------------------------------------------------------------------------
# map_to_delta() — full pipeline
# ---------------------------------------------------------------------------

class TestMapToDelta:
    def test_empty_bundle_returns_empty_delta(self):
        bundle = _make_bundle([])
        provider = _make_provider()
        delta = map_to_delta(bundle, provider)
        assert delta.entries == []

    def test_delta_extraction_bundle_id_set(self):
        entity = _make_entity()
        bundle = _make_bundle([entity])
        provider = _make_provider(_thruster_mapping())
        delta = map_to_delta(bundle, provider)
        assert delta.extraction_bundle_id == bundle.id

    def test_delta_base_ontology_iri_set(self):
        entity = _make_entity()
        bundle = _make_bundle([entity])
        provider = _make_provider(_thruster_mapping())
        delta = map_to_delta(bundle, provider, namespace=_NS)
        assert delta.base_ontology_iri == _NS

    def test_delta_has_entries(self):
        entity = _make_entity()
        bundle = _make_bundle([entity])
        provider = _make_provider(_thruster_mapping())
        delta = map_to_delta(bundle, provider)
        assert len(delta.entries) > 0

    def test_all_entries_proposed(self):
        entity = _make_entity()
        bundle = _make_bundle([entity])
        provider = _make_provider(_thruster_mapping())
        delta = map_to_delta(bundle, provider)
        assert all(e.status == "proposed" for e in delta.entries)

    def test_delta_id_is_16_hex(self):
        bundle = _make_bundle([])
        delta = map_to_delta(bundle, _make_provider())
        assert len(delta.id) == 16

    def test_multiple_entities_all_mapped(self):
        e1 = _make_entity("ThrusterModule_A", "Component")
        e2 = ExtractedEntity(
            id="entity002", text_span="PropellantTank_A",
            chunk_id="c2", source_locator=_make_locator(),
            entity_type="Component", attributes=[],
            extraction_method="llm", confidence=0.9, section_context="",
        )
        bundle = _make_bundle([e1, e2])
        tank_mapping = EntityMappingResponse(
            subject_local_name="PropellantTank_A",
            rdf_type="StorageComponent",
            triples=[],
        )
        provider = _make_provider(_thruster_mapping(), tank_mapping)
        delta = map_to_delta(bundle, provider)
        subjects = {e.triple.subject for e in delta.entries}
        assert _NS + "ThrusterModule_A" in subjects
        assert _NS + "PropellantTank_A" in subjects

    def test_failed_entity_emits_warning_and_continues(self):
        e1 = _make_entity("Alpha", "Component")
        e2 = ExtractedEntity(
            id="entity002", text_span="Beta",
            chunk_id="c2", source_locator=_make_locator(),
            entity_type="Component", attributes=[],
            extraction_method="llm", confidence=0.9, section_context="",
        )
        bundle = _make_bundle([e1, e2])

        call_count = 0

        class PartialProvider:
            provider_name = "pp"
            model_id = "pp"

            def complete(self, request):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("mapper error")
                return LLMResponse(
                    parsed=EntityMappingResponse(
                        subject_local_name="Beta",
                        rdf_type="Component",
                        triples=[],
                    ),
                    raw_json="{}",
                    model_id="pp",
                    usage=TokenUsage(input_tokens=0, output_tokens=0),
                )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            delta = map_to_delta(bundle, PartialProvider())
            assert len(w) == 1
            assert "mapper error" in str(w[0].message)

        # Only Beta's rdf:type triple should be present
        subjects = {e.triple.subject for e in delta.entries}
        assert _NS + "Beta" in subjects
        assert _NS + "Alpha" not in subjects

    def test_approved_entries_pending_after_map(self):
        """map_to_delta returns proposed, not approved."""
        entity = _make_entity()
        bundle = _make_bundle([entity])
        provider = _make_provider(_thruster_mapping())
        delta = map_to_delta(bundle, provider)
        assert delta.approved_entries() == []
        assert len(delta.pending_entries()) > 0
