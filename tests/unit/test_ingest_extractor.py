"""
tests/unit/test_ingest_extractor.py — Tests for ontograph/ingest/extractor.py

All LLM calls are replaced with a MockProvider that returns pre-built
ChunkExtractionResponse objects.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pytest

from ontograph.ingest.extractor import (
    ChunkExtractionResponse,
    RawAttribute,
    RawEntity,
    RawRelationship,
    _build_extraction_messages,
    _convert_entity,
    _make_entity_id,
    extract,
)
from ontograph.llm.base import LLMResponse, TokenUsage
from ontograph.models.document import (
    Chunk,
    DocumentArtifact,
    HeadingNode,
    SourceLocator,
)


# ---------------------------------------------------------------------------
# Helpers: fixtures
# ---------------------------------------------------------------------------

def _make_locator() -> SourceLocator:
    return SourceLocator(
        source_id="abc123",
        source_path="/fake/doc.md",
        source_format="md",
        line_start=1,
        line_end=5,
    )


def _make_chunk(text: str, chunk_id: str = "chunk001") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source_locator=_make_locator(),
        char_start=0,
        char_end=len(text),
        token_count=len(text) // 4,
        section_path=[HeadingNode(level=1, title="Propulsion", anchor="propulsion")],
    )


def _make_artifact(chunks: list[Chunk]) -> DocumentArtifact:
    sha = hashlib.sha256("fake".encode()).hexdigest()
    return DocumentArtifact(
        id="artifact001",
        raw_document_id="raw001",
        source_path="/fake/doc.md",
        source_sha256=sha,
        chunks=chunks,
        created_at="2025-01-01T00:00:00+00:00",
    )


def _make_provider(*responses: ChunkExtractionResponse):
    """Return a mock LLMProvider that yields *responses* in order."""
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


def _thruster_response() -> ChunkExtractionResponse:
    return ChunkExtractionResponse(entities=[
        RawEntity(
            text_span="ThrusterModule_A",
            entity_type="Component",
            confidence=0.95,
            attributes=[
                RawAttribute(name="dry_mass", raw_text="12.4 kg",
                             value="12.4", unit="kg"),
                RawAttribute(name="vacuum_thrust", raw_text="220 N",
                             value="220", unit="N"),
            ],
        )
    ])


# ---------------------------------------------------------------------------
# _make_entity_id
# ---------------------------------------------------------------------------

class TestMakeEntityId:
    def test_returns_16_hex(self):
        eid = _make_entity_id("chunk1", "ThrusterModule_A")
        assert len(eid) == 16
        assert eid.isalnum()

    def test_deterministic(self):
        assert _make_entity_id("c1", "Foo") == _make_entity_id("c1", "Foo")

    def test_different_inputs_differ(self):
        assert _make_entity_id("c1", "Foo") != _make_entity_id("c1", "Bar")


# ---------------------------------------------------------------------------
# _build_extraction_messages
# ---------------------------------------------------------------------------

class TestBuildExtractionMessages:
    def test_two_messages_system_user(self):
        chunk = _make_chunk("The thruster has a mass of 12.4 kg.")
        msgs = _build_extraction_messages(chunk)
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_user_message_contains_section_context(self):
        chunk = _make_chunk("Some text.")
        msgs = _build_extraction_messages(chunk)
        assert "Propulsion" in msgs[1].content

    def test_user_message_contains_chunk_text(self):
        chunk = _make_chunk("The thruster has a mass of 12.4 kg.")
        msgs = _build_extraction_messages(chunk)
        assert "12.4 kg" in msgs[1].content


# ---------------------------------------------------------------------------
# _convert_entity
# ---------------------------------------------------------------------------

class TestConvertEntity:
    def test_entity_id_matches_make_entity_id(self):
        chunk = _make_chunk("text", chunk_id="c1")
        raw = RawEntity(text_span="Foo", entity_type="Component",
                        confidence=0.9, attributes=[])
        e = _convert_entity(raw, chunk)
        assert e.id == _make_entity_id("c1", "Foo")

    def test_extraction_method_is_llm(self):
        chunk = _make_chunk("text")
        raw = RawEntity(text_span="Foo", entity_type="Component",
                        confidence=0.8, attributes=[])
        e = _convert_entity(raw, chunk)
        assert e.extraction_method == "llm"

    def test_entity_type_preserved(self):
        chunk = _make_chunk("text")
        raw = RawEntity(text_span="Foo", entity_type="Subsystem",
                        confidence=0.8, attributes=[])
        e = _convert_entity(raw, chunk)
        assert e.entity_type == "Subsystem"

    def test_section_context_copied(self):
        chunk = _make_chunk("text")
        raw = RawEntity(text_span="Foo", entity_type="Component",
                        confidence=0.8, attributes=[])
        e = _convert_entity(raw, chunk)
        assert "Propulsion" in e.section_context

    def test_attributes_converted(self):
        chunk = _make_chunk("text")
        raw = RawEntity(
            text_span="Foo",
            entity_type="Component",
            confidence=0.9,
            attributes=[
                RawAttribute(name="mass", raw_text="5 kg", value="5", unit="kg")
            ],
        )
        e = _convert_entity(raw, chunk)
        assert len(e.attributes) == 1
        assert e.attributes[0].name == "mass"
        assert e.attributes[0].value == "5"
        assert e.attributes[0].unit == "kg"

    def test_attribute_chunk_id_set(self):
        chunk = _make_chunk("text", chunk_id="chk99")
        raw = RawEntity(
            text_span="Foo", entity_type="Component", confidence=0.9,
            attributes=[RawAttribute(name="x", raw_text="1", value="1", unit=None)],
        )
        e = _convert_entity(raw, chunk)
        assert e.attributes[0].chunk_id == "chk99"

    def test_relationships_converted(self):
        chunk = _make_chunk("text")
        raw = RawEntity(
            text_span="OBC1",
            entity_type="Component",
            confidence=0.9,
            attributes=[],
            relationships=[
                RawRelationship(
                    predicate="usesBusProtocol",
                    target="I2C",
                    raw_text="communicates via I2C",
                ),
                RawRelationship(
                    predicate="usesBusProtocol",
                    target="SPI",
                    raw_text="also uses SPI",
                ),
            ],
        )
        e = _convert_entity(raw, chunk)
        assert len(e.relationships) == 2
        predicates = [r.predicate for r in e.relationships]
        targets = [r.target for r in e.relationships]
        assert predicates == ["usesBusProtocol", "usesBusProtocol"]
        assert "I2C" in targets
        assert "SPI" in targets

    def test_relationship_chunk_id_set(self):
        chunk = _make_chunk("text", chunk_id="chk77")
        raw = RawEntity(
            text_span="OBC1", entity_type="Component", confidence=0.9,
            attributes=[],
            relationships=[
                RawRelationship(predicate="connectsTo", target="EPS", raw_text="connects to EPS")
            ],
        )
        e = _convert_entity(raw, chunk)
        assert e.relationships[0].chunk_id == "chk77"

    def test_no_relationships_by_default(self):
        chunk = _make_chunk("text")
        raw = RawEntity(text_span="Foo", entity_type="Component", confidence=0.9, attributes=[])
        e = _convert_entity(raw, chunk)
        assert e.relationships == []

    def test_system_prompt_mentions_relationships(self):
        from ontograph.ingest.extractor import _build_system_prompt
        prompt = _build_system_prompt(tbox=None)
        assert "relationship" in prompt.lower()


# ---------------------------------------------------------------------------
# extract() — full pipeline with mock provider
# ---------------------------------------------------------------------------

class TestExtract:
    def test_empty_artifact_returns_empty_bundle(self):
        art = _make_artifact([])
        provider = _make_provider()
        bundle = extract(art, provider)
        assert bundle.entities == []

    def test_short_chunk_skipped(self):
        """Chunks below min_chunk_chars should not trigger an LLM call."""
        chunk = _make_chunk("Hi")   # 2 chars < default 50
        art = _make_artifact([chunk])
        # Provider raises if called — proves the chunk was skipped
        class BombProvider:
            provider_name = "bomb"
            model_id = "bomb"
            def complete(self, r):
                raise AssertionError("Should not be called")

        bundle = extract(art, BombProvider(), min_chunk_chars=50)
        assert bundle.entities == []

    def test_entities_extracted_from_chunk(self):
        chunk = _make_chunk("The ThrusterModule_A has a dry mass of 12.4 kg.")
        art = _make_artifact([chunk])
        provider = _make_provider(_thruster_response())
        bundle = extract(art, provider)
        assert len(bundle.entities) == 1
        assert bundle.entities[0].text_span == "ThrusterModule_A"

    def test_bundle_document_artifact_id_set(self):
        chunk = _make_chunk("The ThrusterModule_A has a dry mass of 12.4 kg.")
        art = _make_artifact([chunk])
        provider = _make_provider(_thruster_response())
        bundle = extract(art, provider)
        assert bundle.document_artifact_id == art.id

    def test_bundle_extractor_version_contains_model(self):
        chunk = _make_chunk("The ThrusterModule_A has a mass of 12.4 kg.")
        art = _make_artifact([chunk])
        provider = _make_provider(_thruster_response())
        bundle = extract(art, provider)
        assert "mock-model" in bundle.extractor_version

    def test_multiple_chunks_multiple_calls(self):
        c1 = _make_chunk("Chunk one content here - some component.", "c1")
        c2 = _make_chunk("Chunk two content here - another item.", "c2")
        art = _make_artifact([c1, c2])
        response1 = ChunkExtractionResponse(entities=[
            RawEntity(text_span="Alpha", entity_type="Component",
                      confidence=0.9, attributes=[]),
        ])
        response2 = ChunkExtractionResponse(entities=[
            RawEntity(text_span="Beta", entity_type="Subsystem",
                      confidence=0.8, attributes=[]),
        ])
        provider = _make_provider(response1, response2)
        bundle = extract(art, provider)
        spans = {e.text_span for e in bundle.entities}
        assert "Alpha" in spans
        assert "Beta" in spans

    def test_failed_chunk_emits_warning_and_continues(self):
        c1 = _make_chunk("First chunk with enough text to process.", "c1")
        c2 = _make_chunk("Second chunk with enough text to process.", "c2")
        art = _make_artifact([c1, c2])

        call_count = 0

        class PartialFailProvider:
            provider_name = "partial"
            model_id = "partial"

            def complete(self, r):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("simulated LLM error")
                return LLMResponse(
                    parsed=ChunkExtractionResponse(entities=[
                        RawEntity(text_span="Good", entity_type="Component",
                                  confidence=0.9, attributes=[]),
                    ]),
                    raw_json="{}",
                    model_id="partial",
                    usage=TokenUsage(input_tokens=0, output_tokens=0),
                )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bundle = extract(art, PartialFailProvider())
            assert len(w) == 1
            assert "simulated LLM error" in str(w[0].message)

        assert len(bundle.entities) == 1
        assert bundle.entities[0].text_span == "Good"

    def test_bundle_id_is_16_hex(self):
        chunk = _make_chunk("The ThrusterModule_A has a mass of 12.4 kg.")
        art = _make_artifact([chunk])
        provider = _make_provider(_thruster_response())
        bundle = extract(art, provider)
        assert len(bundle.id) == 16

    def test_entity_ids_unique_across_chunks(self):
        """Same entity name in different chunks should get different IDs."""
        c1 = _make_chunk("ThrusterModule_A info here in chunk one.", "chunk-a")
        c2 = _make_chunk("ThrusterModule_A info here in chunk two.", "chunk-b")
        art = _make_artifact([c1, c2])
        same = RawEntity(text_span="ThrusterModule_A", entity_type="Component",
                         confidence=0.9, attributes=[])
        provider = _make_provider(
            ChunkExtractionResponse(entities=[same]),
            ChunkExtractionResponse(entities=[same]),
        )
        bundle = extract(art, provider)
        ids = [e.id for e in bundle.entities]
        assert ids[0] != ids[1]
