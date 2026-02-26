"""
tests/unit/test_rules_loader.py — Tests for ontograph/rules/loader.py

No LLM calls, no OWL files needed — only YAML parsing.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from ontograph.rules.loader import load_rules
from ontograph.rules.schema import OrgRule, RuleWhen


# ---------------------------------------------------------------------------
# YAML fixtures
# ---------------------------------------------------------------------------

_SINGLE_RULE = textwrap.dedent("""\
    namespace: "http://example.org/aerospace#"

    rules:
      - id: "compat-001"
        name: "Thruster-Tank Mass Compatibility"
        subject_type: "PropulsionSubsystem"
        object_type:  "PropellantTank"
        when:
          attribute: "hasDryMass"
          operator:  ">"
          value:     25.0
          unit:      "kg"
        consequence: "notCompatibleWith"
        severity:    "critical"
        note: "Valve geometry mismatch"
        plain_english: ""
""")

_TWO_RULES = textwrap.dedent("""\
    namespace: "http://example.org/aerospace#"

    rules:
      - id: "compat-001"
        name: "Mass Compat"
        subject_type: "PropulsionSubsystem"
        object_type:  "PropellantTank"
        when:
          attribute: "hasDryMass"
          operator:  ">"
          value:     25.0
          unit:      "kg"
        severity: "critical"

      - id: "compat-002"
        name: "Power Budget"
        subject_type: "PowerSubsystem"
        object_type:  "SolarPanel"
        when:
          attribute: "hasPowerOutput"
          operator:  "<"
          value:     10.0
          unit:      "W"
        severity: "warning"
""")

_SINGLE_ENTITY_RULE = textwrap.dedent("""\
    namespace: "http://example.org/aerospace#"

    rules:
      - id: "single-001"
        name: "Frequency Limit"
        subject_type: "CommunicationSubsystem"
        object_type:  null
        when:
          attribute: "hasOperatingFrequency"
          operator:  ">"
          value:     2400.0
          unit:      "MHz"
        severity: "info"
""")

_NO_UNIT_RULE = textwrap.dedent("""\
    namespace: "http://example.org/aerospace#"

    rules:
      - id: "compat-003"
        name: "Version Check"
        subject_type: "SoftwareComponent"
        object_type:  null
        when:
          attribute: "hasVersion"
          operator:  "=="
          value:     "2.0"
        severity: "warning"
""")

_EMPTY_YAML = ""

_MISSING_REQUIRED_FIELD = textwrap.dedent("""\
    namespace: "http://example.org/aerospace#"

    rules:
      - id: "bad-001"
        name: "Missing when"
        subject_type: "X"
        # 'when' block is missing
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

class TestLoadRules:
    def test_returns_list_of_orgrule(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rules = load_rules(path)
        assert isinstance(rules, list)
        assert len(rules) == 1
        assert isinstance(rules[0], OrgRule)

    def test_empty_yaml_returns_empty_list(self, tmp_path):
        path = _write(tmp_path, "empty.yaml", _EMPTY_YAML)
        rules = load_rules(path)
        assert rules == []

    def test_two_rules_loaded(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _TWO_RULES)
        rules = load_rules(path)
        assert len(rules) == 2

    def test_id_preserved(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rule = load_rules(path)[0]
        assert rule.id == "compat-001"

    def test_name_preserved(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rule = load_rules(path)[0]
        assert rule.name == "Thruster-Tank Mass Compatibility"

    def test_namespace_propagated(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rule = load_rules(path)[0]
        assert rule.namespace == "http://example.org/aerospace#"

    def test_subject_type_set(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rule = load_rules(path)[0]
        assert rule.subject_type == "PropulsionSubsystem"

    def test_object_type_set(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rule = load_rules(path)[0]
        assert rule.object_type == "PropellantTank"

    def test_severity_preserved(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rule = load_rules(path)[0]
        assert rule.severity == "critical"

    def test_note_preserved(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rule = load_rules(path)[0]
        assert "Valve geometry" in rule.note

    def test_plain_english_defaults_to_empty(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rule = load_rules(path)[0]
        assert rule.plain_english == ""


# ---------------------------------------------------------------------------
# RuleWhen fields
# ---------------------------------------------------------------------------

class TestRuleWhen:
    def test_attribute_preserved(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        when = load_rules(path)[0].when
        assert when.attribute == "hasDryMass"

    def test_operator_preserved(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        when = load_rules(path)[0].when
        assert when.operator == ">"

    def test_value_preserved(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        when = load_rules(path)[0].when
        assert when.value == 25.0

    def test_unit_preserved(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        when = load_rules(path)[0].when
        assert when.unit == "kg"

    def test_unit_none_when_absent(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _NO_UNIT_RULE)
        when = load_rules(path)[0].when
        assert when.unit is None

    def test_string_value_preserved(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _NO_UNIT_RULE)
        when = load_rules(path)[0].when
        assert when.value == "2.0"

    def test_less_than_operator(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _TWO_RULES)
        rules = load_rules(path)
        assert rules[1].when.operator == "<"


# ---------------------------------------------------------------------------
# Single-entity rules (object_type=None)
# ---------------------------------------------------------------------------

class TestSingleEntityRule:
    def test_object_type_is_none(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_ENTITY_RULE)
        rule = load_rules(path)[0]
        assert rule.object_type is None

    def test_subject_type_still_set(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_ENTITY_RULE)
        rule = load_rules(path)[0]
        assert rule.subject_type == "CommunicationSubsystem"

    def test_severity_info(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_ENTITY_RULE)
        rule = load_rules(path)[0]
        assert rule.severity == "info"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestLoadRulesErrors:
    def test_missing_when_raises_key_error(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _MISSING_REQUIRED_FIELD)
        with pytest.raises(KeyError):
            load_rules(path)

    def test_invalid_yaml_raises_error(self, tmp_path):
        path = _write(tmp_path, "bad.yaml", "rules:\n  - [invalid yaml {")
        with pytest.raises(yaml.YAMLError):
            load_rules(path)

    def test_accepts_path_as_string(self, tmp_path):
        path = _write(tmp_path, "rules.yaml", _SINGLE_RULE)
        rules = load_rules(str(path))
        assert len(rules) == 1
