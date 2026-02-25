"""
Unit tests for ontograph/utils/owl.py — owl_to_delta() and helpers.

All tests use in-memory rdflib Graphs; no file I/O.
"""

from __future__ import annotations

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from ontograph.utils.owl import owl_to_delta


# ---------------------------------------------------------------------------
# Helpers — build small in-memory graphs
# ---------------------------------------------------------------------------

EX = "http://example.org/test#"


def _make_graph() -> Graph:
    """
    Build a minimal test graph with:

    TBox:
      ex:Widget   rdf:type owl:Class
      ex:massKg   rdf:type owl:DatatypeProperty
      ex:partOf   rdf:type owl:ObjectProperty

    ABox:
      ex:WidgetA  rdf:type ex:Widget
      ex:WidgetA  ex:massKg  "42.0"^^xsd:float
      ex:WidgetA  rdfs:label "Widget A"@en
      ex:WidgetB  rdf:type ex:Widget
      ex:WidgetB  ex:partOf ex:WidgetA
    """
    g = Graph()

    # TBox
    g.add((URIRef(EX + "Widget"),  RDF.type, OWL.Class))
    g.add((URIRef(EX + "massKg"),  RDF.type, OWL.DatatypeProperty))
    g.add((URIRef(EX + "partOf"),  RDF.type, OWL.ObjectProperty))

    # ABox — WidgetA
    wa = URIRef(EX + "WidgetA")
    g.add((wa, RDF.type,             URIRef(EX + "Widget")))
    g.add((wa, URIRef(EX + "massKg"), Literal("42.0", datatype=XSD.float)))
    g.add((wa, RDFS.label,            Literal("Widget A", lang="en")))

    # ABox — WidgetB
    wb = URIRef(EX + "WidgetB")
    g.add((wb, RDF.type,             URIRef(EX + "Widget")))
    g.add((wb, URIRef(EX + "partOf"), wa))

    return g


def _make_bnode_graph() -> Graph:
    """Graph with a blank-node subject (should be skipped)."""
    g = Graph()
    g.add((URIRef(EX + "Thing"), RDF.type, OWL.Class))
    bn = BNode()
    g.add((bn, RDF.type, URIRef(EX + "Thing")))
    g.add((bn, RDFS.label, Literal("unnamed")))
    return g


def _make_named_individual_graph() -> Graph:
    """Graph using owl:NamedIndividual declaration."""
    g = Graph()
    g.add((URIRef(EX + "Device"), RDF.type, OWL.Class))
    ind = URIRef(EX + "DeviceX")
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, URIRef(EX + "Device")))
    g.add((ind, RDFS.label, Literal("Device X")))
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOwlToDelta:

    def test_returns_ontology_delta(self):
        from ontograph.models.ontology import OntologyDelta
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        assert isinstance(delta, OntologyDelta)

    def test_delta_id_and_base_iri(self):
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="my-delta", base_iri=EX)
        assert delta.id == "my-delta"
        assert delta.base_ontology_iri == EX
        assert delta.extraction_bundle_id == "my-delta-bundle"

    def test_all_entries_approved(self):
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        assert len(delta.entries) > 0
        for entry in delta.entries:
            assert entry.status == "approved"
            assert entry.confidence == 1.0

    def test_tbox_classes_not_in_entries(self):
        """owl:Class declarations should not become delta entries."""
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        subjects = {e.triple.subject for e in delta.entries}
        # TBox subjects must not appear as individual subjects
        assert EX + "Widget" not in subjects
        assert EX + "massKg" not in subjects
        assert EX + "partOf"  not in subjects

    def test_individuals_are_subjects(self):
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        subjects = {e.triple.subject for e in delta.entries}
        assert EX + "WidgetA" in subjects
        assert EX + "WidgetB" in subjects

    def test_typed_literal_datatype(self):
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        mass_entries = [
            e for e in delta.entries
            if e.triple.predicate == EX + "massKg"
        ]
        assert len(mass_entries) == 1
        assert mass_entries[0].triple.object == "42.0"
        assert mass_entries[0].triple.datatype == str(XSD.float)

    def test_language_tagged_literal(self):
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        label_entries = [
            e for e in delta.entries
            if e.triple.predicate == str(RDFS.label)
        ]
        assert len(label_entries) >= 1
        wa_label = next(
            e for e in label_entries
            if e.triple.subject == EX + "WidgetA"
        )
        assert wa_label.triple.object == "Widget A"
        assert wa_label.triple.language == "en"
        assert wa_label.triple.datatype is None

    def test_object_property_triple(self):
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        part_entries = [
            e for e in delta.entries
            if e.triple.predicate == EX + "partOf"
        ]
        assert len(part_entries) == 1
        assert part_entries[0].triple.object == EX + "WidgetA"
        assert part_entries[0].triple.datatype is None

    def test_blank_node_subjects_skipped(self):
        g = _make_bnode_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        # No named individuals — delta should have no entries
        assert delta.entries == []

    def test_named_individual_detected(self):
        g = _make_named_individual_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        subjects = {e.triple.subject for e in delta.entries}
        assert EX + "DeviceX" in subjects

    def test_entry_ids_are_unique(self):
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        ids = [e.id for e in delta.entries]
        assert len(ids) == len(set(ids)), "Entry IDs must be unique"

    def test_entry_ids_are_16_chars(self):
        g = _make_graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        for entry in delta.entries:
            assert len(entry.id) == 16

    def test_empty_graph_returns_empty_delta(self):
        g = Graph()
        delta = owl_to_delta(g, delta_id="empty", base_iri=EX)
        assert delta.entries == []

    def test_created_at_default(self):
        g = Graph()
        delta = owl_to_delta(g, delta_id="test", base_iri=EX)
        assert delta.created_at  # non-empty ISO timestamp

    def test_created_at_custom(self):
        g = Graph()
        ts = "2025-01-01T00:00:00+00:00"
        delta = owl_to_delta(g, delta_id="test", base_iri=EX, created_at=ts)
        assert delta.created_at == ts
