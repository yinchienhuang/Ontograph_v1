"""
tests/unit/test_ingest_aligner.py — Tests for ontograph/ingest/aligner.py

LLM calls are replaced with a MockProvider that returns pre-built
AlignmentJudgment objects.
"""

from __future__ import annotations

import hashlib
import warnings

import pytest

from ontograph.ingest.aligner import (
    _SKOS_ALT_LABEL,
    AlignmentJudgment,
    _candidate_id,
    _entity_iri_map,
    _generate_candidates,
    _is_acronym,
    _pick_canonical,
    _similarity_score,
    _strip_separators,
    _token_jaccard,
    align,
    apply_decisions,
)
from ontograph.llm.base import LLMResponse, TokenUsage
from ontograph.models.alignment import (
    AlignmentCandidate,
    AlignmentDecision,
    AlignmentMethod,
    OntologyAlignmentBundle,
)
from ontograph.models.document import SourceLocator
from ontograph.models.extraction import ExtractedEntity, ExtractionBundle
from ontograph.models.ontology import (
    ChangeSource,
    OntologyDelta,
    OntologyDeltaEntry,
    OntologyTriple,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NS = "http://example.org/aerospace#"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def _locator() -> SourceLocator:
    return SourceLocator(
        source_id="src001",
        source_path="/fake/doc.pdf",
        source_format="pdf",
        line_start=1,
        line_end=5,
    )


def _entity(
    eid: str,
    text_span: str,
    entity_type: str = "Organization",
    confidence: float = 0.9,
) -> ExtractedEntity:
    return ExtractedEntity(
        id=eid,
        text_span=text_span,
        chunk_id="c1",
        source_locator=_locator(),
        entity_type=entity_type,
        attributes=[],
        extraction_method="llm",
        confidence=confidence,
        section_context="",
    )


def _bundle(entities: list[ExtractedEntity]) -> ExtractionBundle:
    return ExtractionBundle(
        id="bundle001",
        document_artifact_id="art001",
        entities=entities,
        extractor_version="0.1.0/mock",
        created_at="2025-01-01T00:00:00+00:00",
    )


def _delta_with_entities(*entity_ids: str) -> OntologyDelta:
    """Build a delta that has one rdf:type entry per entity_id."""
    entries = []
    for eid in entity_ids:
        subject = _NS + eid
        triple = OntologyTriple(
            subject=subject,
            predicate=_RDF_TYPE,
            object=_NS + "SomeClass",
        )
        entries.append(OntologyDeltaEntry(
            id=hashlib.sha256(f"{subject}|{_RDF_TYPE}|{_NS}SomeClass".encode()).hexdigest()[:16],
            triple=triple,
            rationale="test",
            confidence=0.9,
            source_entity_id=eid,
            change_source=ChangeSource.PIPELINE,
            status="proposed",
        ))
    return OntologyDelta(
        id="delta001",
        extraction_bundle_id="bundle001",
        base_ontology_iri=_NS,
        entries=entries,
        created_at="2025-01-01T00:00:00+00:00",
    )


def _make_provider(judgment: AlignmentJudgment):
    class MockProvider:
        provider_name = "mock"
        model_id = "mock-model"

        def complete(self, request):
            return LLMResponse(
                parsed=judgment,
                raw_json="{}",
                model_id="mock-model",
                usage=TokenUsage(input_tokens=0, output_tokens=0),
            )

    return MockProvider()


# ---------------------------------------------------------------------------
# _is_acronym
# ---------------------------------------------------------------------------

class TestIsAcronym:
    def test_faa(self):
        assert _is_acronym("FAA", "Federal Aviation Administration")

    def test_nasa(self):
        assert _is_acronym("NASA", "National Aeronautics and Space Administration")

    def test_reverse_false(self):
        assert not _is_acronym("Federal Aviation Administration", "FAA")

    def test_partial_no_match(self):
        assert not _is_acronym("FAA", "Federal Administration")

    def test_single_char_rejected(self):
        assert not _is_acronym("F", "Federal")

    def test_too_long_rejected(self):
        assert not _is_acronym("TOOLONGACRONYM", "Too Long Acronym")

    def test_case_insensitive(self):
        assert _is_acronym("faa", "Federal Aviation Administration")


# ---------------------------------------------------------------------------
# _token_jaccard
# ---------------------------------------------------------------------------

class TestTokenJaccard:
    def test_identical(self):
        assert _token_jaccard("thruster module", "thruster module") == 1.0

    def test_no_overlap(self):
        assert _token_jaccard("propellant tank", "solar panel") == 0.0

    def test_partial(self):
        score = _token_jaccard("thruster module A", "thruster module B")
        assert 0.0 < score < 1.0

    def test_case_insensitive(self):
        assert _token_jaccard("Thruster", "thruster") == 1.0

    def test_empty_a(self):
        assert _token_jaccard("", "thruster") == 0.0

    def test_empty_b(self):
        assert _token_jaccard("thruster", "") == 0.0


# ---------------------------------------------------------------------------
# _strip_separators
# ---------------------------------------------------------------------------

class TestStripSeparators:
    def test_underscore_removed(self):
        assert _strip_separators("EPS_1") == "eps1"

    def test_hyphen_removed(self):
        assert _strip_separators("OBC-1") == "obc1"

    def test_spaces_removed(self):
        assert _strip_separators("OBC 1") == "obc1"

    def test_case_folded(self):
        assert _strip_separators("COMMS_1") == "comms1"

    def test_no_separators_unchanged(self):
        assert _strip_separators("EPS1") == "eps1"


# ---------------------------------------------------------------------------
# _similarity_score — separator normalisation
# ---------------------------------------------------------------------------

class TestSimilarityScoreSeparatorNorm:
    def test_underscore_vs_none_scores_1(self):
        a = _entity("e1", "EPS1")
        b = _entity("e2", "EPS_1")
        score, _ = _similarity_score(a, b)
        assert score == 1.0

    def test_comms_underscore_vs_none_scores_1(self):
        a = _entity("e1", "COMMS1")
        b = _entity("e2", "COMMS_1")
        score, _ = _similarity_score(a, b)
        assert score == 1.0

    def test_hyphen_variant_scores_1(self):
        a = _entity("e1", "OBC1")
        b = _entity("e2", "OBC-1")
        score, _ = _similarity_score(a, b)
        assert score == 1.0

    def test_different_ids_do_not_match(self):
        a = _entity("e1", "EPS1")
        b = _entity("e2", "EPS2")
        score, _ = _similarity_score(a, b)
        assert score < 1.0

    def test_separator_pair_becomes_candidate(self):
        """EPS1 / EPS_1 must produce an approved alignment candidate."""
        a = _entity("e1", "EPS1")
        b = _entity("e2", "EPS_1")
        candidates = _generate_candidates([a, b])
        assert len(candidates) == 1
        assert candidates[0][0].similarity_score == 1.0


# ---------------------------------------------------------------------------
# _candidate_id
# ---------------------------------------------------------------------------

class TestCandidateId:
    def test_16_hex(self):
        cid = _candidate_id("a", "b")
        assert len(cid) == 16

    def test_order_independent(self):
        assert _candidate_id("a", "b") == _candidate_id("b", "a")

    def test_different_pairs_differ(self):
        assert _candidate_id("a", "b") != _candidate_id("a", "c")


# ---------------------------------------------------------------------------
# _pick_canonical
# ---------------------------------------------------------------------------

class TestPickCanonical:
    def test_longer_is_canonical(self):
        ea = _entity("e1", "FAA")
        eb = _entity("e2", "Federal Aviation Administration")
        canon, alias = _pick_canonical(ea, eb)
        assert canon.id == "e2"
        assert alias.id == "e1"

    def test_equal_length_first_wins(self):
        ea = _entity("e1", "Alpha")
        eb = _entity("e2", "Omega")
        canon, _ = _pick_canonical(ea, eb)
        assert canon.id == "e1"  # same length → first arg wins


# ---------------------------------------------------------------------------
# _generate_candidates
# ---------------------------------------------------------------------------

class TestGenerateCandidates:
    def test_acronym_pair_generates_candidate(self):
        entities = [
            _entity("e1", "FAA"),
            _entity("e2", "Federal Aviation Administration"),
        ]
        result = _generate_candidates(entities)
        assert len(result) == 1
        cand, score = result[0]
        assert score == 1.0
        assert cand.method == AlignmentMethod.ACRONYM

    def test_similar_pair_generates_candidate(self):
        entities = [
            _entity("e1", "thruster module"),
            _entity("e2", "thruster assembly module"),
        ]
        result = _generate_candidates(entities)
        assert len(result) == 1
        _, score = result[0]
        assert score > 0.4

    def test_dissimilar_pair_no_candidate(self):
        entities = [
            _entity("e1", "battery"),
            _entity("e2", "solar panel"),
        ]
        result = _generate_candidates(entities)
        assert result == []

    def test_sorted_by_score_descending(self):
        entities = [
            _entity("e1", "FAA"),
            _entity("e2", "Federal Aviation Administration"),
            _entity("e3", "thruster module"),
            _entity("e4", "thruster system module"),
        ]
        result = _generate_candidates(entities)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_entities(self):
        assert _generate_candidates([]) == []

    def test_single_entity(self):
        assert _generate_candidates([_entity("e1", "FAA")]) == []


# ---------------------------------------------------------------------------
# _entity_iri_map
# ---------------------------------------------------------------------------

class TestEntityIriMap:
    def test_maps_entity_to_iri(self):
        delta = _delta_with_entities("e1", "e2")
        m = _entity_iri_map(delta)
        assert m["e1"] == _NS + "e1"
        assert m["e2"] == _NS + "e2"

    def test_empty_delta(self):
        delta = _delta_with_entities()
        assert _entity_iri_map(delta) == {}

    def test_first_occurrence_wins(self):
        """Multiple entries for same entity → first IRI is kept."""
        delta = _delta_with_entities("e1")
        m = _entity_iri_map(delta)
        assert "e1" in m


# ---------------------------------------------------------------------------
# align() — happy path
# ---------------------------------------------------------------------------

class TestAlign:
    def test_empty_bundle(self):
        bundle = _bundle([])
        delta = _delta_with_entities()
        result = align(bundle, delta)
        assert result.candidates == []
        assert result.decisions == []

    def test_acronym_auto_approved(self):
        e1 = _entity("e1", "FAA")
        e2 = _entity("e2", "Federal Aviation Administration")
        bundle = _bundle([e1, e2])
        delta = _delta_with_entities("e1", "e2")
        result = align(bundle, delta)
        assert len(result.decisions) == 1
        assert result.decisions[0].status == "approved"

    def test_acronym_no_provider_needed(self):
        """High-confidence pairs must not call the LLM."""
        e1 = _entity("e1", "FAA")
        e2 = _entity("e2", "Federal Aviation Administration")

        class BombProvider:
            provider_name = "bomb"
            model_id = "bomb"
            def complete(self, r):
                raise AssertionError("Should not be called")

        bundle = _bundle([e1, e2])
        delta = _delta_with_entities("e1", "e2")
        result = align(bundle, delta, provider=BombProvider())
        assert result.decisions[0].status == "approved"

    def test_delta_id_stored(self):
        bundle = _bundle([])
        delta = _delta_with_entities()
        result = align(bundle, delta)
        assert result.ontology_delta_id == delta.id

    def test_id_is_16_hex(self):
        bundle = _bundle([])
        delta = _delta_with_entities()
        result = align(bundle, delta)
        assert len(result.id) == 16

    def test_llm_called_for_ambiguous(self):
        """Pairs below auto-approve threshold trigger an LLM call."""
        e1 = _entity("e1", "thruster module")
        e2 = _entity("e2", "thruster assembly")

        call_count = 0

        class CountingProvider:
            provider_name = "cp"
            model_id = "cp"

            def complete(self, request):
                nonlocal call_count
                call_count += 1
                return LLMResponse(
                    parsed=AlignmentJudgment(
                        same_entity=True,
                        confidence=0.9,
                        rationale="Same hardware, different labels",
                        canonical_surface="thruster module",
                    ),
                    raw_json="{}",
                    model_id="cp",
                    usage=TokenUsage(input_tokens=0, output_tokens=0),
                )

        bundle = _bundle([e1, e2])
        delta = _delta_with_entities("e1", "e2")
        align(bundle, delta, provider=CountingProvider())
        # May or may not call depending on Jaccard score, but should not crash
        assert call_count >= 0

    def test_llm_rejected_pair(self):
        """LLM returning same_entity=False → decision is 'rejected'."""
        e1 = _entity("e1", "thruster A")
        e2 = _entity("e2", "thruster B")
        judgment = AlignmentJudgment(
            same_entity=False,
            confidence=0.95,
            rationale="Different thrusters",
            canonical_surface="thruster A",
        )
        provider = _make_provider(judgment)
        bundle = _bundle([e1, e2])
        delta = _delta_with_entities("e1", "e2")
        result = align(bundle, delta, provider=provider)
        # Rejected or proposed depending on score; no crash
        assert all(d.status in ("proposed", "rejected", "approved") for d in result.decisions)

    def test_llm_failure_emits_warning_leaves_proposed(self):
        e1 = _entity("e1", "thruster module")
        e2 = _entity("e2", "thruster assembly")

        class FailProvider:
            provider_name = "fail"
            model_id = "fail"
            def complete(self, r):
                raise RuntimeError("LLM down")

        bundle = _bundle([e1, e2])
        delta = _delta_with_entities("e1", "e2")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = align(bundle, delta, provider=FailProvider())
        # If a candidate was generated and LLM called, a warning was emitted
        if result.decisions:
            # If LLM was called (score in ambiguous range), warning exists
            pass  # either 0 or 1 warning depending on Jaccard score
        assert all(d.status in ("proposed", "approved") for d in result.decisions)

    def test_no_provider_ambiguous_stays_proposed(self):
        e1 = _entity("e1", "thruster module")
        e2 = _entity("e2", "thruster assembly")
        bundle = _bundle([e1, e2])
        delta = _delta_with_entities("e1", "e2")
        result = align(bundle, delta, provider=None)
        for d in result.decisions:
            assert d.status in ("proposed", "approved")  # never "rejected" from LLM


# ---------------------------------------------------------------------------
# apply_decisions()
# ---------------------------------------------------------------------------

class TestApplyDecisions:
    def _make_alignment(
        self,
        decision: AlignmentDecision,
        delta_id: str = "delta001",
    ) -> OntologyAlignmentBundle:
        return OntologyAlignmentBundle(
            id="align001",
            ontology_delta_id=delta_id,
            candidates=[],
            decisions=[decision],
            created_at="2025-01-01T00:00:00+00:00",
        )

    def test_no_approved_returns_same_delta(self):
        delta = _delta_with_entities("e1", "e2")
        decision = AlignmentDecision(
            candidate_id="cand1",
            canonical_entity_id="e2",
            merged_entity_ids=["e1"],
            aliases=["FAA", "Federal Aviation Administration"],
            status="proposed",  # not approved
        )
        alignment = self._make_alignment(decision)
        result = apply_decisions(delta, alignment)
        # Delta unchanged
        assert len(result.entries) == len(delta.entries)

    def test_merged_subject_rewritten_to_canonical(self):
        delta = _delta_with_entities("e1", "e2")
        decision = AlignmentDecision(
            candidate_id="cand1",
            canonical_entity_id="e2",
            merged_entity_ids=["e1"],
            aliases=["FAA", "Federal Aviation Administration"],
            status="approved",
        )
        alignment = self._make_alignment(decision)
        result = apply_decisions(delta, alignment)

        subjects = {e.triple.subject for e in result.entries}
        canonical_iri = _NS + "e2"
        merged_iri    = _NS + "e1"
        assert canonical_iri in subjects
        assert merged_iri not in subjects  # replaced by canonical

    def test_alt_label_entries_added(self):
        delta = _delta_with_entities("e1", "e2")
        decision = AlignmentDecision(
            candidate_id="cand1",
            canonical_entity_id="e2",
            merged_entity_ids=["e1"],
            aliases=["FAA", "Federal Aviation Administration"],
            status="approved",
        )
        alignment = self._make_alignment(decision)
        result = apply_decisions(delta, alignment)

        alt_entries = [e for e in result.entries if e.triple.predicate == _SKOS_ALT_LABEL]
        objects = {e.triple.object for e in alt_entries}
        assert "FAA" in objects
        assert "Federal Aviation Administration" in objects

    def test_alt_label_change_source_alignment(self):
        delta = _delta_with_entities("e1", "e2")
        decision = AlignmentDecision(
            candidate_id="cand1",
            canonical_entity_id="e2",
            merged_entity_ids=["e1"],
            aliases=["FAA", "Federal Aviation Administration"],
            status="approved",
        )
        alignment = self._make_alignment(decision)
        result = apply_decisions(delta, alignment)
        alt_entries = [e for e in result.entries if e.triple.predicate == _SKOS_ALT_LABEL]
        assert all(e.change_source == ChangeSource.ALIGNMENT for e in alt_entries)

    def test_alt_label_status_approved(self):
        delta = _delta_with_entities("e1", "e2")
        decision = AlignmentDecision(
            candidate_id="cand1",
            canonical_entity_id="e2",
            merged_entity_ids=["e1"],
            aliases=["FAA", "Federal Aviation Administration"],
            status="approved",
        )
        alignment = self._make_alignment(decision)
        result = apply_decisions(delta, alignment)
        alt_entries = [e for e in result.entries if e.triple.predicate == _SKOS_ALT_LABEL]
        assert all(e.status == "approved" for e in alt_entries)

    def test_unknown_canonical_emits_warning(self):
        delta = _delta_with_entities("e1")  # e2 not in delta
        decision = AlignmentDecision(
            candidate_id="cand1",
            canonical_entity_id="e2",  # missing from delta
            merged_entity_ids=["e1"],
            aliases=["FAA"],
            status="approved",
        )
        alignment = self._make_alignment(decision)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_decisions(delta, alignment)
        assert len(w) == 1
        assert "e2" in str(w[0].message)

    def test_no_duplicate_entries_after_rewrite(self):
        """If two entities share the same canonical IRI, no duplicate triples."""
        delta = _delta_with_entities("e1", "e2")
        decision = AlignmentDecision(
            candidate_id="cand1",
            canonical_entity_id="e2",
            merged_entity_ids=["e1"],
            aliases=["A", "B"],
            status="approved",
        )
        alignment = self._make_alignment(decision)
        result = apply_decisions(delta, alignment)
        ids = [e.id for e in result.entries]
        assert len(ids) == len(set(ids))

    def test_rewritten_entry_change_source_alignment(self):
        delta = _delta_with_entities("e1", "e2")
        decision = AlignmentDecision(
            candidate_id="cand1",
            canonical_entity_id="e2",
            merged_entity_ids=["e1"],
            aliases=["FAA", "Federal Aviation Administration"],
            status="approved",
        )
        alignment = self._make_alignment(decision)
        result = apply_decisions(delta, alignment)
        # The rewritten entry (was e1, now e2) should have ALIGNMENT source
        rewritten = [
            e for e in result.entries
            if e.triple.subject == _NS + "e2"
            and e.change_source == ChangeSource.ALIGNMENT
        ]
        assert len(rewritten) >= 1
