"""
tests/unit/test_evaluator.py — Tests for ontograph/evaluator/

All tests use small in-memory Turtle graphs written to temp files so no real
OWL file is needed.  RDF/XML is the default format for the pipeline but Turtle
is far easier to write by hand, so tests pass fmt="turtle" to evaluate().
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ontograph.evaluator import EvaluationReport, evaluate
from ontograph.evaluator.comparator import _abox_individuals, _f1, _subject_triples
from ontograph.utils.owl import load_graph


# ---------------------------------------------------------------------------
# Turtle snippets
# ---------------------------------------------------------------------------

_PREFIXES = textwrap.dedent("""\
    @prefix owl:  <http://www.w3.org/2002/07/owl#> .
    @prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix ex:   <http://test.org/ex#> .
    @prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
""")

# Two individuals, each with one datatype property
_TWO_INDIVIDUALS = _PREFIXES + textwrap.dedent("""\
    ex:ThrusterA a ex:Component ;
        ex:hasDryMass "12.4"^^xsd:float .
    ex:ThrusterB a ex:Component ;
        ex:hasDryMass "8.0"^^xsd:float .
""")

# Only ThrusterA — B is missing
_ONE_INDIVIDUAL = _PREFIXES + textwrap.dedent("""\
    ex:ThrusterA a ex:Component ;
        ex:hasDryMass "12.4"^^xsd:float .
""")

# ThrusterA recovered but with wrong mass value
_ONE_INDIVIDUAL_WRONG_VALUE = _PREFIXES + textwrap.dedent("""\
    ex:ThrusterA a ex:Component ;
        ex:hasDryMass "99.9"^^xsd:float .
""")

# ThrusterA + invented ThrusterC (not in source)
_ONE_PLUS_EXTRA = _PREFIXES + textwrap.dedent("""\
    ex:ThrusterA a ex:Component ;
        ex:hasDryMass "12.4"^^xsd:float .
    ex:ThrusterC a ex:Component ;
        ex:hasDryMass "5.0"^^xsd:float .
""")

# Empty graph (no individuals)
_EMPTY = _PREFIXES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _eval(source_ttl: str, working_ttl: str, tmp_path: Path) -> EvaluationReport:
    src = _write(tmp_path, "source.ttl", source_ttl)
    wrk = _write(tmp_path, "working.ttl", working_ttl)
    return evaluate(src, wrk, fmt="turtle")


# ---------------------------------------------------------------------------
# _f1 helper
# ---------------------------------------------------------------------------

class TestF1:
    def test_perfect(self):
        assert _f1(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_recall(self):
        assert _f1(1.0, 0.0) == pytest.approx(0.0)

    def test_zero_precision(self):
        assert _f1(0.0, 1.0) == pytest.approx(0.0)

    def test_both_zero(self):
        assert _f1(0.0, 0.0) == pytest.approx(0.0)

    def test_typical(self):
        # P=0.8, R=0.6 → F1 = 2*0.48/1.4 ≈ 0.6857
        assert _f1(0.8, 0.6) == pytest.approx(2 * 0.8 * 0.6 / (0.8 + 0.6))


# ---------------------------------------------------------------------------
# _abox_individuals
# ---------------------------------------------------------------------------

class TestAboxIndividuals:
    def test_detects_two_individuals(self, tmp_path):
        g = load_graph(_write(tmp_path, "t.ttl", _TWO_INDIVIDUALS), fmt="turtle")
        inds = _abox_individuals(g)
        assert "http://test.org/ex#ThrusterA" in inds
        assert "http://test.org/ex#ThrusterB" in inds

    def test_empty_graph(self, tmp_path):
        g = load_graph(_write(tmp_path, "t.ttl", _EMPTY), fmt="turtle")
        assert _abox_individuals(g) == set()

    def test_count(self, tmp_path):
        g = load_graph(_write(tmp_path, "t.ttl", _TWO_INDIVIDUALS), fmt="turtle")
        assert len(_abox_individuals(g)) == 2


# ---------------------------------------------------------------------------
# _subject_triples
# ---------------------------------------------------------------------------

class TestSubjectTriples:
    def test_returns_triples_for_subject(self, tmp_path):
        g = load_graph(_write(tmp_path, "t.ttl", _TWO_INDIVIDUALS), fmt="turtle")
        inds = _abox_individuals(g)
        triples = _subject_triples(g, inds)
        # Each individual has rdf:type + hasDryMass = 2 triples each
        assert len(triples) == 4

    def test_empty_individuals_gives_empty(self, tmp_path):
        g = load_graph(_write(tmp_path, "t.ttl", _TWO_INDIVIDUALS), fmt="turtle")
        assert _subject_triples(g, set()) == set()

    def test_triple_is_string_tuple(self, tmp_path):
        g = load_graph(_write(tmp_path, "t.ttl", _ONE_INDIVIDUAL), fmt="turtle")
        inds = _abox_individuals(g)
        for triple in _subject_triples(g, inds):
            assert isinstance(triple, tuple)
            assert all(isinstance(x, str) for x in triple)


# ---------------------------------------------------------------------------
# evaluate() — individual metrics
# ---------------------------------------------------------------------------

class TestEvaluateIndividuals:
    def test_perfect_recall_and_precision(self, tmp_path):
        report = _eval(_TWO_INDIVIDUALS, _TWO_INDIVIDUALS, tmp_path)
        m = report.individuals
        assert m.source_count == 2
        assert m.working_count == 2
        assert m.recovered_count == 2
        assert m.missed_count == 0
        assert m.extra_count == 0
        assert m.recall == pytest.approx(1.0)
        assert m.precision == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_partial_recall_one_missed(self, tmp_path):
        report = _eval(_TWO_INDIVIDUALS, _ONE_INDIVIDUAL, tmp_path)
        m = report.individuals
        assert m.source_count == 2
        assert m.recovered_count == 1
        assert m.missed_count == 1
        assert m.extra_count == 0
        assert m.recall == pytest.approx(0.5)
        assert m.precision == pytest.approx(1.0)
        assert m.f1 == pytest.approx(2 / 3)

    def test_extra_individual_in_working(self, tmp_path):
        report = _eval(_ONE_INDIVIDUAL, _ONE_PLUS_EXTRA, tmp_path)
        m = report.individuals
        assert m.recovered_count == 1
        assert m.extra_count == 1
        assert m.missed_count == 0
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(1.0)

    def test_no_overlap_zero_recall(self, tmp_path):
        no_overlap = _PREFIXES + "ex:ThrusterX a ex:Component .\n"
        report = _eval(_ONE_INDIVIDUAL, no_overlap, tmp_path)
        assert report.individuals.recall == pytest.approx(0.0)
        assert report.individuals.recovered_count == 0

    def test_empty_source(self, tmp_path):
        report = _eval(_EMPTY, _ONE_INDIVIDUAL, tmp_path)
        m = report.individuals
        assert m.source_count == 0
        assert m.recall == pytest.approx(0.0)
        assert m.precision == pytest.approx(0.0)
        assert m.f1 == pytest.approx(0.0)

    def test_empty_working(self, tmp_path):
        report = _eval(_TWO_INDIVIDUALS, _EMPTY, tmp_path)
        m = report.individuals
        assert m.working_count == 0
        assert m.recall == pytest.approx(0.0)

    def test_missed_individuals_list(self, tmp_path):
        report = _eval(_TWO_INDIVIDUALS, _ONE_INDIVIDUAL, tmp_path)
        assert len(report.missed_individuals) == 1
        assert "ThrusterB" in report.missed_individuals[0]

    def test_extra_individuals_list(self, tmp_path):
        report = _eval(_ONE_INDIVIDUAL, _ONE_PLUS_EXTRA, tmp_path)
        assert len(report.extra_individuals) == 1
        assert "ThrusterC" in report.extra_individuals[0]

    def test_no_missed_no_extra_lists_empty(self, tmp_path):
        report = _eval(_TWO_INDIVIDUALS, _TWO_INDIVIDUALS, tmp_path)
        assert report.missed_individuals == []
        assert report.extra_individuals == []


# ---------------------------------------------------------------------------
# evaluate() — triple metrics
# ---------------------------------------------------------------------------

class TestEvaluateTriples:
    def test_perfect_triple_match(self, tmp_path):
        report = _eval(_TWO_INDIVIDUALS, _TWO_INDIVIDUALS, tmp_path)
        m = report.triples
        assert m.recovered_count == m.source_count
        assert m.missed_count == 0
        assert m.extra_count == 0
        assert m.recall == pytest.approx(1.0)
        assert m.precision == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_wrong_value_reduces_triple_recall(self, tmp_path):
        # Individual IRI matches but mass value is wrong → rdf:type recovered, mass missed
        report = _eval(_ONE_INDIVIDUAL, _ONE_INDIVIDUAL_WRONG_VALUE, tmp_path)
        m = report.triples
        # rdf:type triple matches; hasDryMass differs
        assert m.recovered_count >= 1          # rdf:type still matches
        assert m.missed_count >= 1             # hasDryMass "12.4" is missed
        assert m.recall < 1.0

    def test_one_individual_missed_reduces_triple_recall(self, tmp_path):
        report = _eval(_TWO_INDIVIDUALS, _ONE_INDIVIDUAL, tmp_path)
        m = report.triples
        # ThrusterB's triples are missed
        assert m.missed_count > 0
        assert m.recall < 1.0

    def test_empty_source_gives_zero_scores(self, tmp_path):
        report = _eval(_EMPTY, _TWO_INDIVIDUALS, tmp_path)
        m = report.triples
        assert m.source_count == 0
        assert m.recall == pytest.approx(0.0)
        assert m.f1 == pytest.approx(0.0)

    def test_extra_triples_in_working_reduce_precision(self, tmp_path):
        report = _eval(_ONE_INDIVIDUAL, _ONE_PLUS_EXTRA, tmp_path)
        m = report.triples
        assert m.extra_count > 0
        assert m.precision < 1.0


# ---------------------------------------------------------------------------
# evaluate() — report structure
# ---------------------------------------------------------------------------

class TestEvaluateReport:
    def test_returns_evaluation_report(self, tmp_path):
        report = _eval(_TWO_INDIVIDUALS, _TWO_INDIVIDUALS, tmp_path)
        assert isinstance(report, EvaluationReport)

    def test_id_is_16_hex(self, tmp_path):
        report = _eval(_ONE_INDIVIDUAL, _ONE_INDIVIDUAL, tmp_path)
        assert len(report.id) == 16
        assert all(c in "0123456789abcdef" for c in report.id)

    def test_created_at_is_set(self, tmp_path):
        report = _eval(_ONE_INDIVIDUAL, _ONE_INDIVIDUAL, tmp_path)
        assert report.created_at  # non-empty string

    def test_paths_stored(self, tmp_path):
        src = _write(tmp_path, "source.ttl", _ONE_INDIVIDUAL)
        wrk = _write(tmp_path, "working.ttl", _ONE_INDIVIDUAL)
        report = evaluate(src, wrk, fmt="turtle")
        assert "source.ttl" in report.source_owl_path
        assert "working.ttl" in report.working_owl_path

    def test_report_is_json_serialisable(self, tmp_path):
        report = _eval(_TWO_INDIVIDUALS, _ONE_INDIVIDUAL, tmp_path)
        json_str = report.model_dump_json()
        assert "individuals" in json_str
        assert "triples" in json_str

    def test_different_source_working_give_different_ids(self, tmp_path):
        r1 = _eval(_ONE_INDIVIDUAL, _ONE_INDIVIDUAL, tmp_path)
        tmp2 = tmp_path / "sub"
        tmp2.mkdir()
        r2 = _eval(_TWO_INDIVIDUALS, _TWO_INDIVIDUALS, tmp2)
        assert r1.id != r2.id
