"""
Unit tests for ontograph/synthesizer/.

All LLM calls are mocked — no real API access.
Tests cover:
  - label_from_iri
  - build_anchor_map / anchor_to_entry_id
  - _format_triple_for_llm
  - _group_by_subject
  - _build_provenance
  - generate() (mocked provider)
  - self_check: _is_checkable, _values_match, _search_value_in_text, run_self_check
  - attach_self_check
  - format_self_check_report
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ontograph.llm.base import LLMResponse, TokenUsage
from ontograph.models.ontology import (
    ChangeSource,
    OntologyDelta,
    OntologyDeltaEntry,
    OntologyTriple,
)
from ontograph.models.synthesis import SynthesizedDocument
from ontograph.synthesizer.generator import (
    ParagraphDraft,
    SectionDraft,
    _build_provenance,
    _format_triple_for_llm,
    _group_by_subject,
    anchor_to_entry_id,
    build_anchor_map,
    generate,
    label_from_iri,
)
from ontograph.synthesizer.self_check import (
    _is_checkable,
    _search_value_in_text,
    _values_match,
    attach_self_check,
    format_self_check_report,
    run_self_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entry(
    eid: str,
    subject: str,
    predicate: str,
    obj: str,
    datatype: str | None = None,
    status: str = "approved",
) -> OntologyDeltaEntry:
    return OntologyDeltaEntry(
        id=eid,
        triple=OntologyTriple(subject=subject, predicate=predicate, object=obj, datatype=datatype),
        rationale="test",
        confidence=0.9,
        change_source=ChangeSource.PIPELINE,
        status=status,
    )


def _make_delta(entries: list[OntologyDeltaEntry] | None = None) -> OntologyDelta:
    if entries is None:
        entries = [
            _entry("e001", "aero:Thruster_1", "rdf:type",       "aero:Subsystem"),
            _entry("e002", "aero:Thruster_1", "aero:hasMass",   "89",  "xsd:float"),
            _entry("e003", "aero:Thruster_1", "aero:hasIsp",    "310", "xsd:float"),
            _entry("e004", "aero:FuelTank_1", "rdf:type",       "aero:Component"),
            _entry("e005", "aero:FuelTank_1", "aero:hasVolume", "120", "xsd:float"),
        ]
    return OntologyDelta(
        id="delta1",
        extraction_bundle_id="bundle1",
        base_ontology_iri="http://example.org/aero#",
        entries=entries,
        created_at="2025-01-01T00:00:00Z",
    )


def _mock_provider(section_drafts: list[SectionDraft]) -> MagicMock:
    """
    Build a mock LLMProvider that returns section drafts in order.
    Each call to complete() pops the next draft from the list.
    """
    provider = MagicMock()
    responses = [
        LLMResponse(
            parsed=draft,
            raw_json=draft.model_dump_json(),
            model_id="mock-model",
            usage=TokenUsage(input_tokens=100, output_tokens=200),
        )
        for draft in section_drafts
    ]
    provider.complete.side_effect = responses
    return provider


# ---------------------------------------------------------------------------
# label_from_iri
# ---------------------------------------------------------------------------

class TestLabelFromIri:
    def test_prefixed_camel_case(self):
        assert label_from_iri("aero:ThrusterModule") == "Thruster Module"

    def test_prefixed_with_number(self):
        assert label_from_iri("aero:ThrusterModule_42") == "Thruster Module 42"

    def test_full_iri_fragment(self):
        assert label_from_iri("http://example.org/aero#FuelTank") == "Fuel Tank"

    def test_full_iri_path(self):
        assert label_from_iri("http://example.org/aero/PowerBus") == "Power Bus"

    def test_plain_local(self):
        assert label_from_iri("hasMass") == "has Mass"

    def test_rdf_type(self):
        assert label_from_iri("rdf:type") == "type"

    def test_underscore_only(self):
        assert label_from_iri("fuel_tank") == "fuel tank"


# ---------------------------------------------------------------------------
# build_anchor_map
# ---------------------------------------------------------------------------

class TestAnchorMap:
    def test_sequential_anchors(self):
        entries = [_entry(f"e{i:03d}", "s", "p", "o") for i in range(3)]
        amap = build_anchor_map(entries)
        assert len(amap) == 3
        # All anchors are [T-NNN]
        for anchor in amap.values():
            assert anchor.startswith("[T-") and anchor.endswith("]")

    def test_stable_ordering(self):
        """Same entries, different insertion order → same anchor map."""
        e1 = _entry("aaa", "s", "p", "o")
        e2 = _entry("bbb", "s", "p", "o")
        map1 = build_anchor_map([e1, e2])
        map2 = build_anchor_map([e2, e1])
        assert map1 == map2

    def test_roundtrip(self):
        entries = [_entry("e001", "s", "p", "o"), _entry("e002", "s", "p", "o")]
        amap = build_anchor_map(entries)
        rev = anchor_to_entry_id(amap)
        for eid, anchor in amap.items():
            assert rev[anchor] == eid


# ---------------------------------------------------------------------------
# _format_triple_for_llm
# ---------------------------------------------------------------------------

class TestFormatTriple:
    def test_typed_literal(self):
        e = _entry("e001", "aero:Thruster_1", "aero:hasMass", "89", "xsd:float")
        line = _format_triple_for_llm(e, "[T-001]")
        assert "[T-001]" in line
        assert "Thruster 1" in line
        assert "has Mass" in line
        assert '"89"' in line
        assert "float" in line

    def test_object_property(self):
        e = _entry("e002", "aero:Thruster_1", "rdf:type", "aero:Subsystem")
        line = _format_triple_for_llm(e, "[T-002]")
        assert "Subsystem" in line

    def test_plain_string_literal(self):
        e = _entry("e003", "aero:Thruster_1", "rdfs:label", "Main Engine")
        line = _format_triple_for_llm(e, "[T-003]")
        assert '"Main Engine"' in line


# ---------------------------------------------------------------------------
# _group_by_subject
# ---------------------------------------------------------------------------

class TestGroupBySubject:
    def test_groups_correctly(self):
        delta = _make_delta()
        approved = delta.approved_entries()
        groups = _group_by_subject(approved)
        assert "aero:Thruster_1" in groups
        assert "aero:FuelTank_1" in groups
        assert len(groups["aero:Thruster_1"]) == 3
        assert len(groups["aero:FuelTank_1"]) == 2

    def test_single_subject(self):
        entries = [_entry("e1", "aero:X", "p", "o"), _entry("e2", "aero:X", "p2", "o2")]
        groups = _group_by_subject(entries)
        assert list(groups.keys()) == ["aero:X"]


# ---------------------------------------------------------------------------
# _build_provenance
# ---------------------------------------------------------------------------

class TestBuildProvenance:
    def test_basic_provenance(self):
        entries = [_entry("e001", "s", "p", "o"), _entry("e002", "s", "p2", "o2")]
        amap = build_anchor_map(entries)
        rev = anchor_to_entry_id(amap)

        a1 = amap["e001"]
        a2 = amap["e002"]

        draft = SectionDraft(
            section_title="Test Section",
            paragraphs=[
                ParagraphDraft(
                    text=f"Fact one {a1}. Fact two {a2}.",
                    cited_anchors=[a1, a2],
                )
            ],
        )
        provenance = _build_provenance([("s", draft)], rev)
        assert len(provenance) == 1
        assert set(provenance[0].triple_ids) == {"e001", "e002"}

    def test_paragraph_index_offset(self):
        draft = SectionDraft(
            section_title="S",
            paragraphs=[ParagraphDraft(text="t", cited_anchors=[])],
        )
        prov = _build_provenance([("s", draft)], {}, para_offset=5)
        assert prov[0].paragraph_index == 5

    def test_anchors_from_cited_anchors_field_even_if_not_in_text(self):
        entries = [_entry("e001", "s", "p", "o")]
        amap = build_anchor_map(entries)
        rev = anchor_to_entry_id(amap)
        a1 = amap["e001"]

        draft = SectionDraft(
            section_title="S",
            paragraphs=[
                ParagraphDraft(text="No anchor in text.", cited_anchors=[a1])
            ],
        )
        prov = _build_provenance([("s", draft)], rev)
        assert "e001" in prov[0].triple_ids


# ---------------------------------------------------------------------------
# generate() — end-to-end with mocked provider
# ---------------------------------------------------------------------------

class TestGenerate:
    def _make_sections(self, delta: OntologyDelta) -> list[SectionDraft]:
        approved = delta.approved_entries()
        amap = build_anchor_map(approved)
        groups = _group_by_subject(approved)

        sections = []
        for subject_iri, entries in groups.items():
            label = label_from_iri(subject_iri)
            paragraphs = []
            for e in entries:
                anchor = amap[e.id]
                paragraphs.append(ParagraphDraft(
                    text=f"The {label_from_iri(e.triple.predicate)} is "
                         f"{e.triple.object} {anchor}.",
                    cited_anchors=[anchor],
                ))
            sections.append(SectionDraft(
                section_title=f"{label} Overview",
                paragraphs=paragraphs,
            ))
        return sections

    def test_generate_returns_synthesized_document(self):
        delta = _make_delta()
        sections = self._make_sections(delta)
        provider = _mock_provider(sections)

        doc = generate(delta, provider, title="Test Report")

        assert isinstance(doc, SynthesizedDocument)
        assert doc.title == "Test Report"
        assert doc.ontology_delta_id == "delta1"
        assert doc.self_check is None  # not run yet

    def test_generate_calls_provider_once_per_subject(self):
        delta = _make_delta()
        sections = self._make_sections(delta)
        provider = _mock_provider(sections)

        generate(delta, provider, title="T")

        # 2 subjects → 2 LLM calls
        assert provider.complete.call_count == 2

    def test_generate_markdown_contains_title(self):
        delta = _make_delta()
        sections = self._make_sections(delta)
        provider = _mock_provider(sections)

        doc = generate(delta, provider, title="Propulsion CONOPS")

        assert "# Propulsion CONOPS" in doc.markdown

    def test_generate_markdown_contains_section_headings(self):
        delta = _make_delta()
        sections = self._make_sections(delta)
        provider = _mock_provider(sections)

        doc = generate(delta, provider, title="T")

        assert "## Thruster 1 Overview" in doc.markdown
        assert "## Fuel Tank 1 Overview" in doc.markdown

    def test_generate_provenance_non_empty(self):
        delta = _make_delta()
        sections = self._make_sections(delta)
        provider = _mock_provider(sections)

        doc = generate(delta, provider, title="T")

        assert len(doc.provenance) > 0

    def test_generate_raises_on_empty_delta(self):
        delta = _make_delta(entries=[])
        provider = MagicMock()

        with pytest.raises(ValueError, match="no approved entries"):
            generate(delta, provider)

    def test_generate_raises_if_all_proposed(self):
        entries = [_entry("e1", "s", "p", "o", status="proposed")]
        delta = _make_delta(entries=entries)
        provider = MagicMock()

        with pytest.raises(ValueError, match="no approved entries"):
            generate(delta, provider)


# ---------------------------------------------------------------------------
# self_check — _is_checkable
# ---------------------------------------------------------------------------

class TestIsCheckable:
    def test_typed_literal_is_checkable(self):
        e = _entry("e1", "s", "aero:hasMass", "89", "xsd:float")
        assert _is_checkable(e)

    def test_rdf_type_not_checkable(self):
        e = _entry("e1", "s", "rdf:type", "aero:Subsystem")
        assert not _is_checkable(e)

    def test_object_iri_not_checkable(self):
        e = _entry("e1", "s", "aero:hasInterface", "http://example.org/aero#Interface_1")
        assert not _is_checkable(e)

    def test_plain_string_literal_checkable(self):
        e = _entry("e1", "s", "rdfs:label", "Main Thruster")
        assert _is_checkable(e)

    def test_prefixed_type_not_checkable(self):
        e = _entry("e1", "s", "rdf:type", "owl:Class")
        assert not _is_checkable(e)


# ---------------------------------------------------------------------------
# self_check — _search_value_in_text
# ---------------------------------------------------------------------------

class TestSearchValueInText:
    def test_exact_match(self):
        assert _search_value_in_text("89 kg", "The mass is 89 kg total.") is not None

    def test_numeric_with_tolerance(self):
        # 89.5 is within 1% of 89 → should match
        result = _search_value_in_text("89", "The mass is approximately 89.5 kg.")
        assert result is not None

    def test_not_found(self):
        assert _search_value_in_text("89", "No numbers here.") is None

    def test_string_literal(self):
        result = _search_value_in_text("Main Engine", "Component: Main Engine assembly.")
        assert result is not None

    def test_numeric_outside_tolerance(self):
        # 100 is >1% from 89
        result = _search_value_in_text("89", "The mass is 100 kg.")
        assert result is None


# ---------------------------------------------------------------------------
# self_check — _values_match
# ---------------------------------------------------------------------------

class TestValuesMatch:
    def test_exact_numeric(self):
        match, note = _values_match("89", "89")
        assert match
        assert note is None

    def test_numeric_within_tolerance(self):
        match, note = _values_match("89", "89.5")
        assert match

    def test_numeric_outside_tolerance(self):
        match, note = _values_match("89", "100")
        assert not match
        assert note is not None

    def test_string_match(self):
        match, _ = _values_match("Main Engine", "Main Engine")
        assert match

    def test_string_case_insensitive(self):
        match, _ = _values_match("main engine", "MAIN ENGINE")
        assert match

    def test_none_found(self):
        match, note = _values_match("89", None)
        assert not match
        assert "not found" in note

    def test_zero_value(self):
        match, _ = _values_match("0", "0.0")
        assert match


# ---------------------------------------------------------------------------
# run_self_check
# ---------------------------------------------------------------------------

class TestRunSelfCheck:
    def _make_doc(self, markdown: str) -> SynthesizedDocument:
        return SynthesizedDocument(
            id="doc1",
            ontology_delta_id="delta1",
            title="T",
            markdown=markdown,
            provenance=[],
            self_check=None,
            created_at="2025-01-01T00:00:00Z",
        )

    def test_all_match(self):
        delta = _make_delta()
        markdown = (
            "# Report\n\n"
            "The thruster has a mass of 89 kg.\n"
            "The specific impulse is 310 s.\n"
            "The tank volume is 120 L.\n"
        )
        doc = self._make_doc(markdown)
        result = run_self_check(doc, delta)

        assert result.coverage == 1.0
        assert result.matched_count == result.checked_triple_count
        assert result.discrepancies == []

    def test_partial_match(self):
        delta = _make_delta()
        # Only mass is present, Isp and volume are missing
        markdown = "# Report\n\nThe thruster has a mass of 89 kg.\n"
        doc = self._make_doc(markdown)
        result = run_self_check(doc, delta)

        assert result.coverage < 1.0
        assert len(result.discrepancies) > 0

    def test_wrong_value(self):
        delta = _make_delta()
        # Wrong mass value (200 is >1% from 89)
        markdown = (
            "# Report\n\nMass is 200 kg. Isp is 310 s. Volume is 120 L.\n"
        )
        doc = self._make_doc(markdown)
        result = run_self_check(doc, delta)

        mass_items = [i for i in result.items if "e002" in i.triple_id or i.expected_object == "89"]
        assert any(not item.match for item in mass_items)

    def test_rdf_type_not_checked(self):
        """rdf:type triples should not appear in self-check items."""
        delta = _make_delta()
        doc = self._make_doc("# T\n\nMass 89. Isp 310. Volume 120.\n")
        result = run_self_check(doc, delta)

        # e001 (rdf:type for Thruster_1) and e004 (rdf:type for FuelTank_1) skipped
        checked_ids = {item.triple_id for item in result.items}
        assert "e001" not in checked_ids
        assert "e004" not in checked_ids

    def test_empty_delta_gives_full_coverage(self):
        delta = OntologyDelta(
            id="d", extraction_bundle_id="b",
            base_ontology_iri="http://x",
            entries=[], created_at="2025-01-01T00:00:00Z",
        )
        doc = self._make_doc("# Empty\n")
        result = run_self_check(doc, delta)
        assert result.coverage == 1.0
        assert result.checked_triple_count == 0


# ---------------------------------------------------------------------------
# attach_self_check
# ---------------------------------------------------------------------------

class TestAttachSelfCheck:
    def test_returns_new_document(self):
        delta = _make_delta()
        doc = SynthesizedDocument(
            id="d1", ontology_delta_id="delta1", title="T",
            markdown="# T\n\n89 310 120\n",
            provenance=[], self_check=None,
            created_at="2025-01-01T00:00:00Z",
        )
        updated = attach_self_check(doc, delta)
        # Original unchanged
        assert doc.self_check is None
        # New document has result
        assert updated.self_check is not None

    def test_self_check_attached(self):
        delta = _make_delta()
        doc = SynthesizedDocument(
            id="d1", ontology_delta_id="delta1", title="T",
            markdown="# T\n\n89 310 120\n",
            provenance=[], self_check=None,
            created_at="2025-01-01T00:00:00Z",
        )
        updated = attach_self_check(doc, delta)
        assert isinstance(updated.self_check.coverage, float)


# ---------------------------------------------------------------------------
# format_self_check_report
# ---------------------------------------------------------------------------

class TestFormatSelfCheckReport:
    def test_all_pass_message(self):
        delta = _make_delta()
        doc = SynthesizedDocument(
            id="d1", ontology_delta_id="delta1", title="T",
            markdown="# T\n\n89 310 120\n",
            provenance=[], self_check=None,
            created_at="2025-01-01T00:00:00Z",
        )
        updated = attach_self_check(doc, delta)
        report = format_self_check_report(updated.self_check)
        assert "verified successfully" in report

    def test_discrepancy_listed(self):
        delta = _make_delta()
        doc = SynthesizedDocument(
            id="d1", ontology_delta_id="delta1", title="T",
            markdown="# T\n\nNo useful numbers here.\n",
            provenance=[], self_check=None,
            created_at="2025-01-01T00:00:00Z",
        )
        updated = attach_self_check(doc, delta)
        report = format_self_check_report(updated.self_check)
        assert "discrepancy" in report
