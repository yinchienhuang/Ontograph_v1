"""
rules/loader.py — Parse a YAML rules file into OrgRule objects.

YAML format:
    namespace: "http://example.org/aerospace#"

    rules:
      - id: "compat-001"
        name: "Thruster-Tank Mass Compatibility"
        subject_type: "PropulsionSubsystem"
        object_type:  "PropellantTank"
        when:
          attribute: "hasDryMass"    # property local name on the object individual
          operator:  ">"
          value:     25.0
          unit:      "kg"
        consequence: "notCompatibleWith"
        severity:    "critical"
        note: "Valve geometry mismatch — thruster mounting cannot support > 25 kg tanks"
        plain_english: ""            # auto-filled by generator if blank
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ontograph.rules.schema import OrgRule, RuleWhen


def load_rules(path: str | Path) -> list[OrgRule]:
    """
    Parse a YAML rules file and return a list of OrgRule objects.

    The top-level ``namespace`` field is propagated to each rule so the checker
    can resolve local attribute names to full OWL property IRIs.

    Args:
        path: Path to a YAML file following the rules schema.

    Returns:
        List of :class:`~ontograph.rules.schema.OrgRule` objects.

    Raises:
        KeyError:        If a rule is missing a required field (``id``, ``name``,
                         ``when.attribute``, ``when.operator``, or ``when.value``).
        yaml.YAMLError:  If the file is not valid YAML.
    """
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return []

    data: dict[str, Any] = raw
    namespace: str = str(data.get("namespace", ""))
    rules_data: list[dict[str, Any]] = data.get("rules") or []

    rules: list[OrgRule] = []
    for r in rules_data:
        when_data: dict[str, Any] = r["when"]
        when = RuleWhen(
            attribute=str(when_data["attribute"]),
            operator=str(when_data["operator"]),
            value=when_data["value"],
            unit=str(when_data["unit"]) if when_data.get("unit") is not None else None,
        )

        rule = OrgRule(
            id=str(r["id"]),
            name=str(r["name"]),
            namespace=namespace,
            subject_type=str(r["subject_type"]) if r.get("subject_type") is not None else None,
            object_type=str(r["object_type"]) if r.get("object_type") is not None else None,
            when=when,
            consequence=str(r.get("consequence") or ""),
            severity=r.get("severity", "warning"),
            note=str(r.get("note") or "").strip(),
            plain_english=str(r.get("plain_english") or "").strip(),
        )
        rules.append(rule)

    return rules
