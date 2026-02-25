"""
Unit tests for all data contract models.

Tests: round-trip JSON serialization, computed fields, helper methods.
No I/O, no LLM calls.
"""

import json

import pytest

from ontograph.models.document import (
    Chunk,
    DocumentArtifact,
    HeadingNode,
    RawDocument,
    SourceLocator,
)
from ontograph.models.extraction import (
    ExtractedAttribute,
    ExtractedEntity,
    ExtractionBundle,
)
from ontograph.models.ontology import (
    ChangeSource,
    OntologyChangelog,
    OntologyDelta,
    OntologyDeltaEntry,
    OntologyTriple,
)
from ontograph.models.alignment import (
    AlignmentCandidate,
    AlignmentDecision,
    AlignmentMethod,
    OntologyAlignmentBundle,
)
from ontograph.models.synthesis import (
    FactCheckItem,
    ParagraphProvenance,
    SelfCheckResult,
    SynthesizedDocument,
)
from ontograph.models.evaluation import (
    ArmMetrics,
    AttributeChange,
    DesignChangeRequest,
    EvaluationArm,
    EvaluationResult,
    EvidenceSnippet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_locator(fmt: str = "pdf") -> SourceLocator:
    return SourceLocator(
        source_id="abc123",
        source_path="/data/raw/doc.pdf",
        source_format=fmt,
        page=3,
    )


def make_chunk(idx: int = 0) -> Chunk:
    loc = make_locator()
    return Chunk(
        id=Chunk.make_id("abc123", idx * 100, idx * 100 + 50),
        text="The thruster has a mass of 89 kg.",
        source_locator=loc,
        char_start=idx * 100,
        char_end=idx * 100 + 50,
        token_count=12,
        section_path=[
            HeadingNode(level=1, title="3. Propulsion", anchor="3-propulsion"),
            HeadingNode(level=2, title="3.2 Thruster Subsystem", anchor="3-2-thruster-subsystem"),
        ],
    )


# ---------------------------------------------------------------------------
# SourceLocator
# ---------------------------------------------------------------------------

class TestSourceLocator:
    def test_pdf_uri(self):
        loc = make_locator("pdf")
        assert loc.to_uri() == "file:///data/raw/doc.pdf#page=3"

    def test_text_uri(self):
        loc = SourceLocator(
            source_id="x", source_path="/data/raw/doc.txt",
            source_format="txt", line_start=42,
        )
        assert loc.to_uri() == "file:///data/raw/doc.txt#L42"

    def test_fallback_uri(self):
        loc = SourceLocator(
            source_id="x", source_path="/data/raw/doc.md",
            source_format="md",
        )
        assert loc.to_uri() == "file:///data/raw/doc.md"


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_section_context(self):
        c = make_chunk()
        assert c.section_context == "3. Propulsion > 3.2 Thruster Subsystem"

    def test_section_context_empty(self):
        loc = make_locator()
        c = Chunk(
            id="x", text="intro", source_locator=loc,
            char_start=0, char_end=5, token_count=1,
        )
        assert c.section_context == "(no section)"

    def test_to_llm_context_includes_breadcrumb(self):
        c = make_chunk()
        ctx = c.to_llm_context()
        assert ctx.startswith("[Section: 3. Propulsion > 3.2 Thruster Subsystem]")
        assert "The thruster has a mass of 89 kg." in ctx

    def test_make_id_deterministic(self):
        id1 = Chunk.make_id("src", 0, 50)
        id2 = Chunk.make_id("src", 0, 50)
        assert id1 == id2

    def test_make_id_differs_on_different_offsets(self):
        assert Chunk.make_id("src", 0, 50) != Chunk.make_id("src", 0, 51)

    def test_json_roundtrip(self):
        c = make_chunk()
        restored = Chunk.model_validate_json(c.model_dump_json())
        assert restored.id == c.id
        assert restored.section_context == c.section_context


# ---------------------------------------------------------------------------
# DocumentArtifact
# ---------------------------------------------------------------------------

class TestDocumentArtifact:
    def test_chunk_by_id(self):
        chunks = [make_chunk(i) for i in range(3)]
        artifact = DocumentArtifact(
            id="art1", raw_document_id="raw1",
            source_path="/p", source_sha256="sha",
            chunks=chunks, created_at="2025-01-01T00:00:00Z",
        )
        found = artifact.chunk_by_id(chunks[1].id)
        assert found is not None
        assert found.id == chunks[1].id

    def test_chunk_by_id_not_found(self):
        artifact = DocumentArtifact(
            id="art1", raw_document_id="raw1",
            source_path="/p", source_sha256="sha",
            chunks=[], created_at="2025-01-01T00:00:00Z",
        )
        assert artifact.chunk_by_id("nonexistent") is None


# ---------------------------------------------------------------------------
# ExtractionBundle
# ---------------------------------------------------------------------------

class TestExtractionBundle:
    def _make_bundle(self) -> ExtractionBundle:
        loc = make_locator()
        attr = ExtractedAttribute(
            name="mass", raw_text="89 kg", value="89", unit="kg",
            chunk_id="c1", source_locator=loc,
        )
        entity = ExtractedEntity(
            id="e1", text_span="ThrusterModule",
            chunk_id="c1", source_locator=loc,
            entity_type="Subsystem",
            attributes=[attr],
            extraction_method="hybrid",
            confidence=0.95,
        )
        return ExtractionBundle(
            id="bundle1", document_artifact_id="art1",
            entities=[entity], extractor_version="0.1.0/claude-sonnet-4-6",
            created_at="2025-01-01T00:00:00Z",
        )

    def test_entities_by_type(self):
        bundle = self._make_bundle()
        assert len(bundle.entities_by_type("Subsystem")) == 1
        assert len(bundle.entities_by_type("Material")) == 0

    def test_entity_by_id(self):
        bundle = self._make_bundle()
        assert bundle.entity_by_id("e1") is not None
        assert bundle.entity_by_id("missing") is None

    def test_json_roundtrip(self):
        bundle = self._make_bundle()
        restored = ExtractionBundle.model_validate_json(bundle.model_dump_json())
        assert restored.id == bundle.id
        assert len(restored.entities) == 1
        assert restored.entities[0].attributes[0].unit == "kg"


# ---------------------------------------------------------------------------
# OntologyDelta
# ---------------------------------------------------------------------------

class TestOntologyDelta:
    def _make_delta(self) -> OntologyDelta:
        triple = OntologyTriple(
            subject="aero:ThrusterModule_1",
            predicate="aero:hasMass",
            object="89",
            datatype="xsd:float",
        )
        entry = OntologyDeltaEntry(
            id="entry1", triple=triple,
            rationale="Extracted from section 3.2",
            confidence=0.9,
            source_entity_id="e1",
            source_chunk_id="c1",
            change_source=ChangeSource.PIPELINE,
        )
        return OntologyDelta(
            id="delta1", extraction_bundle_id="bundle1",
            base_ontology_iri="http://example.org/aerospace#",
            entries=[entry], created_at="2025-01-01T00:00:00Z",
        )

    def test_pending_entries(self):
        delta = self._make_delta()
        assert len(delta.pending_entries()) == 1
        assert len(delta.approved_entries()) == 0

    def test_approve_entry(self):
        delta = self._make_delta()
        delta.entries[0].status = "approved"
        assert len(delta.approved_entries()) == 1
        assert len(delta.pending_entries()) == 0

    def test_entry_by_id(self):
        delta = self._make_delta()
        assert delta.entry_by_id("entry1") is not None
        assert delta.entry_by_id("nope") is None

    def test_json_roundtrip(self):
        delta = self._make_delta()
        restored = OntologyDelta.model_validate_json(delta.model_dump_json())
        assert restored.entries[0].triple.datatype == "xsd:float"


# ---------------------------------------------------------------------------
# OntologyAlignmentBundle
# ---------------------------------------------------------------------------

class TestAlignmentBundle:
    def _make_bundle(self) -> OntologyAlignmentBundle:
        candidate = AlignmentCandidate(
            id="cand1",
            entity_id_a="e1", entity_id_b="e2",
            surface_a="FAA", surface_b="Federal Aviation Administration",
            similarity_score=0.92,
            method=AlignmentMethod.ACRONYM,
            rationale="FAA is a known acronym for Federal Aviation Administration",
        )
        decision = AlignmentDecision(
            candidate_id="cand1",
            canonical_entity_id="e2",
            merged_entity_ids=["e1"],
            aliases=["FAA", "Federal Aviation Administration"],
            status="approved",
        )
        return OntologyAlignmentBundle(
            id="align1", ontology_delta_id="delta1",
            candidates=[candidate], decisions=[decision],
            created_at="2025-01-01T00:00:00Z",
        )

    def test_approved_decisions(self):
        bundle = self._make_bundle()
        assert len(bundle.approved_decisions()) == 1

    def test_pending_decisions(self):
        bundle = self._make_bundle()
        assert len(bundle.pending_decisions()) == 0

    def test_decision_for(self):
        bundle = self._make_bundle()
        assert bundle.decision_for("cand1") is not None
        assert bundle.decision_for("nope") is None


# ---------------------------------------------------------------------------
# SynthesizedDocument
# ---------------------------------------------------------------------------

class TestSynthesizedDocument:
    def _make_doc(self) -> SynthesizedDocument:
        prov = ParagraphProvenance(
            paragraph_index=0,
            triple_ids=["entry1", "entry2"],
            citation_anchors=["[T-001]", "[T-002]"],
        )
        check = SelfCheckResult(
            checked_triple_count=2,
            matched_count=2,
            coverage=1.0,
            items=[
                FactCheckItem(triple_id="entry1", expected_object="89", found_text="89", match=True),
                FactCheckItem(triple_id="entry2", expected_object="310", found_text="310", match=True),
            ],
        )
        return SynthesizedDocument(
            id="syn1", ontology_delta_id="delta1",
            title="Propulsion CONOPS",
            markdown="The thruster has a mass of 89 kg [T-001].",
            provenance=[prov],
            self_check=check,
            created_at="2025-01-01T00:00:00Z",
        )

    def test_triples_cited(self):
        doc = self._make_doc()
        assert doc.triples_cited() == {"entry1", "entry2"}

    def test_is_fully_grounded(self):
        doc = self._make_doc()
        assert doc.is_fully_grounded()

    def test_discrepancies_empty_when_all_match(self):
        doc = self._make_doc()
        assert doc.self_check is not None
        assert doc.self_check.discrepancies == []


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

class TestEvaluationResult:
    def _make_result(self) -> EvaluationResult:
        arm_docs = EvaluationArm(
            name="docs_only",
            predicted_impacted_items=["ThrusterModule", "FuelTank"],
            reasoning_trace="...",
            evidence=[],
        )
        arm_ont = EvaluationArm(
            name="docs_plus_ontology",
            predicted_impacted_items=["ThrusterModule", "FuelTank", "PowerBus"],
            reasoning_trace="...",
            evidence=[],
        )
        return EvaluationResult(
            id="eval1",
            design_change_request_id="dcr1",
            synthesized_document_id="syn1",
            ground_truth_impacted_items=["ThrusterModule", "FuelTank", "PowerBus"],
            arms=[arm_docs, arm_ont],
            metrics={
                "docs_only": ArmMetrics(precision=1.0, recall=0.67, f1=0.80, evidence_quality=0.7),
                "docs_plus_ontology": ArmMetrics(precision=1.0, recall=1.0, f1=1.0, evidence_quality=0.9),
            },
            created_at="2025-01-01T00:00:00Z",
        )

    def test_winner(self):
        result = self._make_result()
        assert result.winner() == "docs_plus_ontology"

    def test_arm_lookup(self):
        result = self._make_result()
        assert result.arm("docs_only") is not None
        assert result.arm("docs_plus_ontology") is not None

    def test_json_roundtrip(self):
        result = self._make_result()
        restored = EvaluationResult.model_validate_json(result.model_dump_json())
        assert restored.winner() == "docs_plus_ontology"
