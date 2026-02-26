"""Unit tests for ontograph/impact/ module."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from rdflib import Graph
from rdflib import Literal as RDFLiteral
from rdflib import URIRef
from rdflib.namespace import OWL, RDF, XSD

from ontograph.impact.analyzer import _apply_change, _score, analyze_impact
from ontograph.impact.loader import load_scenarios
from ontograph.impact.schema import (
    AttributeChangeSpec,
    ImpactArmResult,
    ImpactAnalysisResult,
    ImpactScenario,
)
from ontograph.rules.schema import ViolationInstance, ViolationReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NS = "http://example.org/test#"


def _make_owl(tmp_path, triples: list[tuple]) -> "Path":
    """Create a minimal OWL file with the given triples."""
    from pathlib import Path

    g = Graph()
    for s, p, o in triples:
        g.add((s, p, o))
    path = tmp_path / "test.owl"
    g.serialize(str(path), format="xml")
    return path


def _make_violation_report(
    violations: list[tuple[str, str, bool]],
    mode: str = "both",
) -> ViolationReport:
    """Build a ViolationReport from (rule_id, arm_mode, violated) tuples."""
    instances = [
        ViolationInstance(
            rule_id=rule_id,
            severity="warning",
            mode=arm_mode,  # type: ignore[arg-type]
            subject_label="Comp1",
            violated=violated,
            confidence=1.0,
            explanation="test",
        )
        for rule_id, arm_mode, violated in violations
    ]
    return ViolationReport(
        id="test",
        created_at=datetime.now(timezone.utc).isoformat(),
        rules_file="test.yaml",
        owl_path="test.owl",
        document_path=None,
        mode=mode,
        violations=instances,
    )


# ---------------------------------------------------------------------------
# load_scenarios tests
# ---------------------------------------------------------------------------

def test_load_scenarios_valid(tmp_path):
    yaml_content = (
        'namespace: "http://example.org/test#"\n'
        "scenarios:\n"
        "  - id: s1\n"
        '    description: "A test scenario"\n'
        '    component_local: "Comp1"\n'
        "    attribute_changes:\n"
        "      - property_local: powerW\n"
        "        old_value: '2.0'\n"
        "        new_value: '1.0'\n"
        "        unit: W\n"
        "    ground_truth_violations:\n"
        "      - rule-001\n"
    )
    f = tmp_path / "scenarios.yaml"
    f.write_text(yaml_content)

    ns, scenarios = load_scenarios(f)
    assert ns == "http://example.org/test#"
    assert len(scenarios) == 1
    s = scenarios[0]
    assert s.id == "s1"
    assert s.component_local == "Comp1"
    assert len(s.attribute_changes) == 1
    assert s.attribute_changes[0].property_local == "powerW"
    assert s.attribute_changes[0].new_value == "1.0"
    assert s.attribute_changes[0].unit == "W"
    assert s.ground_truth_violations == ["rule-001"]


def test_load_scenarios_empty_list(tmp_path):
    yaml_content = 'namespace: "http://example.org/test#"\nscenarios: []\n'
    f = tmp_path / "scenarios.yaml"
    f.write_text(yaml_content)
    ns, scenarios = load_scenarios(f)
    assert ns == "http://example.org/test#"
    assert scenarios == []


def test_load_scenarios_missing_field(tmp_path):
    # Missing component_local
    yaml_content = (
        'namespace: "http://example.org/test#"\n'
        "scenarios:\n"
        "  - id: s1\n"
        '    description: "test"\n'
        "    attribute_changes: []\n"
        "    ground_truth_violations: []\n"
    )
    f = tmp_path / "scenarios.yaml"
    f.write_text(yaml_content)
    with pytest.raises(KeyError):
        load_scenarios(f)


def test_load_scenarios_no_namespace(tmp_path):
    yaml_content = (
        "scenarios:\n"
        "  - id: s1\n"
        '    description: "test"\n'
        '    component_local: "Comp1"\n'
        "    attribute_changes: []\n"
        "    ground_truth_violations: []\n"
    )
    f = tmp_path / "scenarios.yaml"
    f.write_text(yaml_content)
    ns, scenarios = load_scenarios(f)
    assert ns == ""


# ---------------------------------------------------------------------------
# _apply_change tests
# ---------------------------------------------------------------------------

def test_apply_change_modifies_triple(tmp_path):
    subj = URIRef(NS + "Comp1")
    pred = URIRef(NS + "powerW")
    owl_path = _make_owl(tmp_path, [
        (subj, pred, RDFLiteral("2.0", datatype=XSD.decimal)),
        (subj, RDF.type, URIRef(NS + "Component")),
    ])

    scenario = ImpactScenario(
        id="s1", description="test", component_local="Comp1",
        attribute_changes=[
            AttributeChangeSpec(property_local="powerW", old_value="2.0", new_value="1.0"),
        ],
        ground_truth_violations=[],
    )

    modified = _apply_change(owl_path, NS, scenario)
    values = list(modified.objects(subj, pred))
    assert len(values) == 1
    assert str(values[0]) == "1.0"


def test_apply_change_preserves_datatype(tmp_path):
    subj = URIRef(NS + "Comp1")
    pred = URIRef(NS + "powerW")
    owl_path = _make_owl(tmp_path, [
        (subj, pred, RDFLiteral("2.0", datatype=XSD.decimal)),
    ])

    scenario = ImpactScenario(
        id="s1", description="test", component_local="Comp1",
        attribute_changes=[
            AttributeChangeSpec(property_local="powerW", old_value="2.0", new_value="1.0"),
        ],
        ground_truth_violations=[],
    )

    modified = _apply_change(owl_path, NS, scenario)
    values = list(modified.objects(URIRef(NS + "Comp1"), URIRef(NS + "powerW")))
    assert values[0].datatype == XSD.decimal


def test_apply_change_preserves_integer_datatype(tmp_path):
    subj = URIRef(NS + "Comp1")
    pred = URIRef(NS + "count")
    owl_path = _make_owl(tmp_path, [
        (subj, pred, RDFLiteral("5", datatype=XSD.integer)),
    ])

    scenario = ImpactScenario(
        id="s1", description="test", component_local="Comp1",
        attribute_changes=[
            AttributeChangeSpec(property_local="count", old_value="5", new_value="10"),
        ],
        ground_truth_violations=[],
    )

    modified = _apply_change(owl_path, NS, scenario)
    values = list(modified.objects(subj, pred))
    assert values[0].datatype == XSD.integer
    assert str(values[0]) == "10"


def test_apply_change_unknown_individual_no_error(tmp_path):
    """Applying a change to an individual that does not exist should not raise."""
    owl_path = _make_owl(tmp_path, [])

    scenario = ImpactScenario(
        id="s1", description="test", component_local="NonExistent",
        attribute_changes=[
            AttributeChangeSpec(property_local="powerW", old_value="2.0", new_value="1.0"),
        ],
        ground_truth_violations=[],
    )
    # Should complete without error
    _apply_change(owl_path, NS, scenario)


def test_apply_change_multiple_attributes(tmp_path):
    subj = URIRef(NS + "Radio1")
    pred_pw = URIRef(NS + "powerW")
    pred_dr = URIRef(NS + "dataRateMbps")
    owl_path = _make_owl(tmp_path, [
        (subj, pred_pw, RDFLiteral("4.0", datatype=XSD.decimal)),
        (subj, pred_dr, RDFLiteral("0.256", datatype=XSD.decimal)),
    ])

    scenario = ImpactScenario(
        id="s1", description="test", component_local="Radio1",
        attribute_changes=[
            AttributeChangeSpec(property_local="powerW",      old_value="4.0",   new_value="2.0"),
            AttributeChangeSpec(property_local="dataRateMbps", old_value="0.256", new_value="2.0"),
        ],
        ground_truth_violations=[],
    )

    modified = _apply_change(owl_path, NS, scenario)
    assert str(list(modified.objects(subj, pred_pw))[0]) == "2.0"
    assert str(list(modified.objects(subj, pred_dr))[0]) == "2.0"


# ---------------------------------------------------------------------------
# _score tests
# ---------------------------------------------------------------------------

def test_score_perfect():
    p, r, f1 = _score({"a", "b"}, {"a", "b"})
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_score_empty_both():
    p, r, f1 = _score(set(), set())
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_score_empty_prediction():
    p, r, f1 = _score(set(), {"a", "b"})
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_score_empty_ground_truth():
    # All predictions are false positives → P=0/1=0, R=0/0=0, F1=0
    p, r, f1 = _score({"a"}, set())
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_score_partial_overlap():
    # predicted={a,b,c}, gt={a,b} → TP=2, P=2/3, R=1.0, F1=0.8
    p, r, f1 = _score({"a", "b", "c"}, {"a", "b"})
    assert abs(p - 2 / 3) < 1e-6
    assert r == 1.0
    assert abs(f1 - 0.8) < 1e-6


def test_score_case_insensitive():
    p, r, f1 = _score({"RULE-001"}, {"rule-001"})
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


# ---------------------------------------------------------------------------
# ImpactAnalysisResult helper tests
# ---------------------------------------------------------------------------

def _make_result(arms: list[ImpactArmResult], winner: str | None) -> ImpactAnalysisResult:
    return ImpactAnalysisResult(
        id="test",
        created_at=datetime.now(timezone.utc).isoformat(),
        scenario_id="s1",
        description="test",
        scenarios_file="test.yaml",
        owl_path="test.owl",
        document_path=None,
        baseline_violations=["rule-001"],
        ground_truth_violations=[],
        arms=arms,
        winner=winner,
    )


def test_arm_lookup_found():
    arms = [
        ImpactArmResult(arm="ontology", violations_after=[], precision=1.0, recall=1.0, f1=1.0),
        ImpactArmResult(arm="document", violations_after=["r1"], precision=0.0, recall=0.0, f1=0.0),
    ]
    r = _make_result(arms, "ontology")
    assert r.arm("ontology") is not None
    assert r.arm("ontology").f1 == 1.0


def test_arm_lookup_missing():
    arms = [
        ImpactArmResult(arm="ontology", violations_after=[], precision=1.0, recall=1.0, f1=1.0),
    ]
    r = _make_result(arms, "ontology")
    assert r.arm("document") is None


def test_winner_ontology():
    arms = [
        ImpactArmResult(arm="ontology", violations_after=[], precision=1.0, recall=1.0, f1=1.0),
        ImpactArmResult(arm="document", violations_after=["r1"], precision=0.0, recall=0.0, f1=0.0),
    ]
    r = _make_result(arms, "ontology")
    assert r.winner == "ontology"


def test_winner_tied():
    arms = [
        ImpactArmResult(arm="ontology", violations_after=["r1"], precision=1.0, recall=0.5, f1=0.667),
        ImpactArmResult(arm="document", violations_after=["r2"], precision=1.0, recall=0.5, f1=0.667),
    ]
    r = _make_result(arms, None)
    assert r.winner is None


# ---------------------------------------------------------------------------
# analyze_impact integration test (mocked check_rules)
# ---------------------------------------------------------------------------

def test_analyze_impact_both_arms(mocker, tmp_path):
    """analyze_impact should call check_rules twice and build correct arm results."""
    subj = URIRef(NS + "OBC1")
    pred = URIRef(NS + "powerW")
    owl_path = _make_owl(tmp_path, [
        (subj, pred, RDFLiteral("2.0", datatype=XSD.decimal)),
        (subj, RDF.type, URIRef(NS + "OnBoardComputer")),
    ])

    # baseline: rule-001 violated in both arms
    baseline_report = _make_violation_report([
        ("rule-001", "ontology", True),
        ("rule-001", "document", True),
    ])
    # post-change: ontology arm correctly detects resolution; document arm stays stale
    post_report = _make_violation_report([
        ("rule-001", "ontology", False),   # correctly resolved in OWL
        ("rule-001", "document", True),    # stale document still says violated
    ])

    mock_check = mocker.patch("ontograph.impact.analyzer.check_rules")
    mock_check.side_effect = [baseline_report, post_report]

    scenario = ImpactScenario(
        id="s1",
        description="OBC1.powerW drops to 1.0W",
        component_local="OBC1",
        attribute_changes=[
            AttributeChangeSpec(property_local="powerW", old_value="2.0", new_value="1.0"),
        ],
        ground_truth_violations=[],   # GT: no violations after change
    )

    result = analyze_impact(
        scenario=scenario,
        namespace=NS,
        rules=[],
        rules_file="test.yaml",
        provider=MagicMock(),
        working_owl=owl_path,
        mode="both",
    )

    assert result.scenario_id == "s1"
    assert result.baseline_violations == ["rule-001"]
    assert result.ground_truth_violations == []

    ont = result.arm("ontology")
    doc = result.arm("document")
    assert ont is not None
    assert doc is not None

    # Ontology arm: detected nothing → correct vs GT={}
    assert ont.violations_after == []
    assert ont.precision == 0.0   # no predictions
    assert ont.recall    == 0.0   # GT empty
    assert ont.f1        == 0.0

    # Document arm: detected rule-001 → wrong (GT is empty, so no TP)
    assert "rule-001" in doc.violations_after
    assert doc.precision == 0.0
    assert doc.recall    == 0.0
    assert doc.f1        == 0.0

    # Both tied at 0.0 → winner is None
    assert result.winner is None


def test_analyze_impact_ontology_wins(mocker, tmp_path):
    """Ontology arm should win when it correctly predicts the post-change violations."""
    subj = URIRef(NS + "OBC1")
    pred = URIRef(NS + "powerW")
    owl_path = _make_owl(tmp_path, [
        (subj, pred, RDFLiteral("2.0", datatype=XSD.decimal)),
    ])

    # baseline: both arms see rule-001 and rule-002 violated
    baseline_report = _make_violation_report([
        ("rule-001", "ontology", True),
        ("rule-002", "ontology", True),
        ("rule-001", "document", True),
        ("rule-002", "document", True),
    ])
    # post-change: ontology arm correctly shows only rule-002 remains
    # document arm still sees both (stale)
    post_report = _make_violation_report([
        ("rule-001", "ontology", False),
        ("rule-002", "ontology", True),
        ("rule-001", "document", True),
        ("rule-002", "document", True),
    ])

    mock_check = mocker.patch("ontograph.impact.analyzer.check_rules")
    mock_check.side_effect = [baseline_report, post_report]

    scenario = ImpactScenario(
        id="s1",
        description="test",
        component_local="OBC1",
        attribute_changes=[
            AttributeChangeSpec(property_local="powerW", old_value="2.0", new_value="1.0"),
        ],
        ground_truth_violations=["rule-002"],   # only rule-002 should remain
    )

    result = analyze_impact(
        scenario=scenario,
        namespace=NS,
        rules=[],
        rules_file="test.yaml",
        provider=MagicMock(),
        working_owl=owl_path,
        mode="both",
    )

    ont = result.arm("ontology")
    doc = result.arm("document")
    assert ont is not None
    assert doc is not None

    # Ontology: detected {rule-002}, GT={rule-002} → P=1 R=1 F1=1
    assert ont.violations_after == ["rule-002"]
    assert ont.precision == 1.0
    assert ont.recall    == 1.0
    assert ont.f1        == 1.0

    # Document: detected {rule-001, rule-002}, GT={rule-002} → P=0.5 R=1 F1=0.667
    assert set(doc.violations_after) == {"rule-001", "rule-002"}
    assert abs(doc.precision - 0.5) < 1e-4
    assert doc.recall == 1.0
    assert abs(doc.f1 - 2/3) < 1e-4

    assert result.winner == "ontology"


def test_analyze_impact_ontology_only_mode(mocker, tmp_path):
    """mode='ontology' should produce only one arm (ontology)."""
    owl_path = _make_owl(tmp_path, [])

    baseline_report = _make_violation_report([("r1", "ontology", True)], mode="ontology")
    post_report     = _make_violation_report([("r1", "ontology", False)], mode="ontology")

    mock_check = mocker.patch("ontograph.impact.analyzer.check_rules")
    mock_check.side_effect = [baseline_report, post_report]

    scenario = ImpactScenario(
        id="s1", description="test", component_local="X",
        attribute_changes=[AttributeChangeSpec(property_local="p", old_value="1", new_value="2")],
        ground_truth_violations=[],
    )

    result = analyze_impact(
        scenario=scenario, namespace=NS, rules=[], rules_file="test.yaml",
        provider=MagicMock(), working_owl=owl_path, mode="ontology",
    )

    assert len(result.arms) == 1
    assert result.arms[0].arm == "ontology"
    # Single arm → winner is always that arm
    assert result.winner == "ontology"
