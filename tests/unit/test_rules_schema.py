"""
tests/unit/test_rules_schema.py — Tests for ontograph/rules/schema.py

No LLM calls, no file I/O needed — pure Pydantic model testing.
"""

from __future__ import annotations

import pytest

from ontograph.rules.schema import (
    OrgRule,
    RuleWhen,
    ViolationInstance,
    ViolationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rule_when(**kwargs) -> RuleWhen:
    defaults = dict(attribute="hasDryMass", operator=">", value=25.0, unit="kg")
    defaults.update(kwargs)
    return RuleWhen(**defaults)


def _make_org_rule(**kwargs) -> OrgRule:
    defaults = dict(
        id="r-001",
        name="Test Rule",
        subject_type="PropulsionSubsystem",
        object_type="PropellantTank",
        when=_make_rule_when(),
    )
    defaults.update(kwargs)
    return OrgRule(**defaults)


def _make_violation(rule_id: str, violated: bool, mode: str = "ontology",
                    severity: str = "warning", subject: str = "SubA",
                    obj: str | None = "ObjB") -> ViolationInstance:
    return ViolationInstance(
        rule_id=rule_id,
        severity=severity,
        mode=mode,
        subject_label=subject,
        object_label=obj,
        violated=violated,
        confidence=0.9,
        explanation="test explanation",
    )


def _make_report(violations: list[ViolationInstance], mode: str = "ontology") -> ViolationReport:
    return ViolationReport(
        id="abc12345678defgh"[:16],
        created_at="2026-01-01T00:00:00+00:00",
        rules_file="test_rules.yaml",
        mode=mode,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# RuleWhen
# ---------------------------------------------------------------------------

class TestRuleWhen:
    def test_basic_construction(self):
        w = _make_rule_when()
        assert w.attribute == "hasDryMass"
        assert w.operator == ">"
        assert w.value == 25.0
        assert w.unit == "kg"

    def test_unit_none_by_default(self):
        w = RuleWhen(attribute="hasVersion", operator="==", value="2.0")
        assert w.unit is None

    def test_string_value(self):
        w = RuleWhen(attribute="hasName", operator="==", value="Alpha")
        assert w.value == "Alpha"

    def test_int_value(self):
        w = RuleWhen(attribute="hasCount", operator=">", value=3)
        assert w.value == 3

    def test_all_operators_accepted(self):
        for op in (">", "<", ">=", "<=", "==", "!="):
            w = RuleWhen(attribute="x", operator=op, value=1.0)
            assert w.operator == op


# ---------------------------------------------------------------------------
# OrgRule
# ---------------------------------------------------------------------------

class TestOrgRule:
    def test_basic_construction(self):
        rule = _make_org_rule()
        assert rule.id == "r-001"
        assert rule.name == "Test Rule"

    def test_null_object_type_single_entity_rule(self):
        rule = _make_org_rule(object_type=None)
        assert rule.object_type is None

    def test_null_subject_type(self):
        rule = _make_org_rule(subject_type=None)
        assert rule.subject_type is None

    def test_severity_default_warning(self):
        rule = _make_org_rule()
        assert rule.severity == "warning"

    def test_severity_critical(self):
        rule = _make_org_rule(severity="critical")
        assert rule.severity == "critical"

    def test_severity_info(self):
        rule = _make_org_rule(severity="info")
        assert rule.severity == "info"

    def test_plain_english_empty_by_default(self):
        rule = _make_org_rule()
        assert rule.plain_english == ""

    def test_plain_english_can_be_set(self):
        rule = _make_org_rule(plain_english="Heavy tanks may be incompatible.")
        assert "Heavy tanks" in rule.plain_english

    def test_note_empty_by_default(self):
        rule = _make_org_rule()
        assert rule.note == ""

    def test_namespace_empty_by_default(self):
        rule = _make_org_rule()
        assert rule.namespace == ""

    def test_namespace_can_be_set(self):
        rule = _make_org_rule(namespace="http://example.org/#")
        assert rule.namespace == "http://example.org/#"

    def test_model_copy_with_plain_english(self):
        rule = _make_org_rule()
        updated = rule.model_copy(update={"plain_english": "Some vague note."})
        assert updated.plain_english == "Some vague note."
        assert rule.plain_english == ""  # original unchanged


# ---------------------------------------------------------------------------
# ViolationInstance
# ---------------------------------------------------------------------------

class TestViolationInstance:
    def test_basic_construction(self):
        vi = _make_violation("r-001", violated=True)
        assert vi.rule_id == "r-001"
        assert vi.violated is True

    def test_violated_false(self):
        vi = _make_violation("r-001", violated=False)
        assert vi.violated is False

    def test_object_label_can_be_none(self):
        vi = _make_violation("r-001", violated=True, obj=None)
        assert vi.object_label is None

    def test_confidence_bounds(self):
        # Should not raise
        ViolationInstance(
            rule_id="r-001", severity="warning", mode="ontology",
            subject_label="S", violated=True, confidence=0.0, explanation=""
        )
        ViolationInstance(
            rule_id="r-001", severity="warning", mode="ontology",
            subject_label="S", violated=True, confidence=1.0, explanation=""
        )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(Exception):
            ViolationInstance(
                rule_id="r-001", severity="warning", mode="ontology",
                subject_label="S", violated=True, confidence=1.5, explanation=""
            )

    def test_mode_ontology(self):
        vi = _make_violation("r-001", violated=True, mode="ontology")
        assert vi.mode == "ontology"

    def test_mode_document(self):
        vi = _make_violation("r-001", violated=True, mode="document")
        assert vi.mode == "document"


# ---------------------------------------------------------------------------
# ViolationReport — critical() and by_rule()
# ---------------------------------------------------------------------------

class TestViolationReport:
    def test_critical_filters_violated_true(self):
        violations = [
            _make_violation("r-001", violated=True),
            _make_violation("r-001", violated=False),
            _make_violation("r-002", violated=True),
        ]
        report = _make_report(violations)
        critical = report.critical()
        assert len(critical) == 2
        assert all(v.violated for v in critical)

    def test_critical_empty_when_none_violated(self):
        violations = [
            _make_violation("r-001", violated=False),
            _make_violation("r-002", violated=False),
        ]
        report = _make_report(violations)
        assert report.critical() == []

    def test_critical_all_when_all_violated(self):
        violations = [_make_violation(f"r-{i}", violated=True) for i in range(3)]
        report = _make_report(violations)
        assert len(report.critical()) == 3

    def test_by_rule_filters_correctly(self):
        violations = [
            _make_violation("r-001", violated=True),
            _make_violation("r-001", violated=False),
            _make_violation("r-002", violated=True),
        ]
        report = _make_report(violations)
        r001 = report.by_rule("r-001")
        assert len(r001) == 2
        assert all(v.rule_id == "r-001" for v in r001)

    def test_by_rule_returns_empty_for_unknown_id(self):
        violations = [_make_violation("r-001", violated=True)]
        report = _make_report(violations)
        assert report.by_rule("does-not-exist") == []

    def test_by_rule_returns_all_modes(self):
        violations = [
            _make_violation("r-001", violated=True, mode="ontology"),
            _make_violation("r-001", violated=True, mode="document"),
        ]
        report = _make_report(violations, mode="both")
        assert len(report.by_rule("r-001")) == 2

    def test_empty_violations_list(self):
        report = _make_report([])
        assert report.critical() == []
        assert report.by_rule("r-001") == []

    def test_json_serialisable(self):
        violations = [_make_violation("r-001", violated=True)]
        report = _make_report(violations)
        json_str = report.model_dump_json()
        assert "violations" in json_str
        assert "r-001" in json_str

    def test_optional_paths_none(self):
        report = _make_report([])
        assert report.owl_path is None
        assert report.document_path is None

    def test_optional_paths_set(self):
        report = ViolationReport(
            id="abcd1234abcd1234",
            created_at="2026-01-01T00:00:00+00:00",
            rules_file="r.yaml",
            owl_path="/some/working.owl",
            document_path="/some/doc.md",
            mode="both",
            violations=[],
        )
        assert report.owl_path == "/some/working.owl"
        assert report.document_path == "/some/doc.md"
