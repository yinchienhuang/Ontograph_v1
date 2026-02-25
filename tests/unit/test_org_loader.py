"""
tests/unit/test_org_loader.py — Tests for ontograph/ingest/org_loader.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ontograph.ingest.org_loader import _resolve_iri, load_org_knowledge
from ontograph.models.ontology import ChangeSource


# ---------------------------------------------------------------------------
# _resolve_iri unit tests
# ---------------------------------------------------------------------------

NS = "http://example.org/aero#"
ALIASES = {
    "notCompatibleWith": "http://example.org/aero#notCompatibleWith",
    "hasMass": "http://example.org/aero#hasMass",
}


class TestResolveIri:
    def test_full_http_iri_passthrough(self):
        iri = "http://example.org/aero#Component"
        assert _resolve_iri(iri, NS, ALIASES) == iri

    def test_full_https_iri_passthrough(self):
        iri = "https://example.org/aero#Component"
        assert _resolve_iri(iri, NS, ALIASES) == iri

    def test_alias_resolves(self):
        assert _resolve_iri("notCompatibleWith", NS, ALIASES) == \
            "http://example.org/aero#notCompatibleWith"

    def test_alias_takes_priority_over_short_name(self):
        # "hasMass" is both an alias and could be expanded as a short name —
        # alias wins
        assert _resolve_iri("hasMass", NS, ALIASES) == \
            "http://example.org/aero#hasMass"

    def test_short_name_expanded_with_namespace(self):
        assert _resolve_iri("ThrusterModule_A", NS, {}) == \
            "http://example.org/aero#ThrusterModule_A"

    def test_short_name_with_colon_passthrough(self):
        # Prefixed notation like xsd:float is left unchanged
        result = _resolve_iri("xsd:float", NS, ALIASES)
        assert result == "xsd:float"

    def test_empty_namespace_short_name(self):
        result = _resolve_iri("Foo", "", {})
        assert result == "Foo"


# ---------------------------------------------------------------------------
# Helpers: write YAML to a temp file
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "org.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_org_knowledge — basic behaviour
# ---------------------------------------------------------------------------

class TestLoadOrgKnowledge:
    def test_basic_two_rules(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "ComponentA"
                predicate: "http://example.org/aero#notCompatibleWith"
                object:    "ComponentB"
                note:      "Test rule"
              - id: "org-002"
                subject:   "ComponentB"
                predicate: "http://example.org/aero#requiresInspectionOf"
                object:    "ComponentC"
        """)
        entries = load_org_knowledge(p)
        assert len(entries) == 2

    def test_change_source_is_manual(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "A"
                predicate: "http://example.org/aero#rel"
                object:    "B"
        """)
        entries = load_org_knowledge(p)
        assert all(e.change_source == ChangeSource.MANUAL for e in entries)

    def test_status_is_approved(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "A"
                predicate: "http://example.org/aero#rel"
                object:    "B"
        """)
        entries = load_org_knowledge(p)
        assert all(e.status == "approved" for e in entries)

    def test_confidence_is_one(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "A"
                predicate: "http://example.org/aero#rel"
                object:    "B"
        """)
        entries = load_org_knowledge(p)
        assert all(e.confidence == 1.0 for e in entries)

    def test_no_source_chunk_or_entity_id(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "A"
                predicate: "http://example.org/aero#rel"
                object:    "B"
        """)
        entries = load_org_knowledge(p)
        assert entries[0].source_chunk_id is None
        assert entries[0].source_entity_id is None

    def test_note_stored_as_rationale(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "A"
                predicate: "http://example.org/aero#rel"
                object:    "B"
                note:      "Important constraint"
        """)
        entries = load_org_knowledge(p)
        assert "Important constraint" in entries[0].rationale

    def test_short_names_expanded_with_namespace(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "ComponentA"
                predicate: "http://example.org/aero#notCompatibleWith"
                object:    "ComponentB"
        """)
        entries = load_org_knowledge(p)
        t = entries[0].triple
        assert t.subject == "http://example.org/aero#ComponentA"
        assert t.object  == "http://example.org/aero#ComponentB"

    def test_full_iris_preserved(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "http://other.org/ns#ComponentX"
                predicate: "http://example.org/aero#rel"
                object:    "http://other.org/ns#ComponentY"
        """)
        entries = load_org_knowledge(p)
        t = entries[0].triple
        assert t.subject == "http://other.org/ns#ComponentX"
        assert t.object  == "http://other.org/ns#ComponentY"

    def test_predicate_alias_resolved(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            predicates:
                notCompatibleWith: "http://example.org/aero#notCompatibleWith"
            rules:
              - id: "org-001"
                subject:   "A"
                predicate: "notCompatibleWith"
                object:    "B"
        """)
        entries = load_org_knowledge(p)
        assert entries[0].triple.predicate == "http://example.org/aero#notCompatibleWith"

    def test_literal_triple_with_datatype(self, tmp_path: Path):
        XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"
        p = _write_yaml(tmp_path, f"""
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "ComponentA"
                predicate: "http://example.org/aero#maxCycles"
                object:    "5000"
                datatype:  "{XSD_INTEGER}"
        """)
        entries = load_org_knowledge(p)
        t = entries[0].triple
        assert t.object == "5000"
        assert t.datatype == XSD_INTEGER

    def test_literal_object_not_expanded_with_namespace(self, tmp_path: Path):
        """When datatype is set the object value must not be IRI-expanded."""
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "ComponentA"
                predicate: "http://example.org/aero#label"
                object:    "some label"
                datatype:  "http://www.w3.org/2001/XMLSchema#string"
        """)
        entries = load_org_knowledge(p)
        assert entries[0].triple.object == "some label"

    def test_empty_rules_returns_empty_list(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules: []
        """)
        assert load_org_knowledge(p) == []

    def test_empty_file_returns_empty_list(self, tmp_path: Path):
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        assert load_org_knowledge(p) == []

    def test_no_namespace_key(self, tmp_path: Path):
        """Rules with full IRIs work even without a namespace key."""
        p = _write_yaml(tmp_path, """
            rules:
              - id: "org-001"
                subject:   "http://example.org/aero#A"
                predicate: "http://example.org/aero#rel"
                object:    "http://example.org/aero#B"
        """)
        entries = load_org_knowledge(p)
        assert len(entries) == 1

    def test_ids_preserved(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "my-custom-id"
                subject:   "A"
                predicate: "http://example.org/aero#rel"
                object:    "B"
        """)
        entries = load_org_knowledge(p)
        assert entries[0].id == "my-custom-id"

    def test_missing_required_field_raises(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """
            namespace: "http://example.org/aero#"
            rules:
              - id: "org-001"
                subject:   "A"
                # predicate is missing
                object:    "B"
        """)
        with pytest.raises(KeyError):
            load_org_knowledge(p)
