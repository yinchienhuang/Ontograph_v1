"""
impact/loader.py — Load impact analysis scenarios from a YAML file.

YAML format::

    namespace: "http://example.org/cubesat-ontology#"
    scenarios:
      - id: "impact-001"
        description: "Replace OBC1 with a power-efficient model"
        component_local: "OBC1"
        attribute_changes:
          - property_local: "powerW"
            old_value: "2.0"
            new_value: "1.0"
            unit: "W"
        ground_truth_violations:
          - "compat-001"
          - "compat-002"
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ontograph.impact.schema import AttributeChangeSpec, ImpactScenario


def load_scenarios(path: Path | str) -> tuple[str, list[ImpactScenario]]:
    """
    Parse an impact-scenarios YAML file.

    Returns:
        A tuple of ``(namespace, list[ImpactScenario])``.

    Raises:
        KeyError:            If a required field is missing from an entry.
        yaml.YAMLError:      If the file is not valid YAML.
        FileNotFoundError:   If the file does not exist.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    namespace: str = raw.get("namespace", "")

    scenarios: list[ImpactScenario] = []
    for entry in raw.get("scenarios", []):
        changes = [
            AttributeChangeSpec(
                property_local=c["property_local"],
                old_value=str(c["old_value"]),
                new_value=str(c["new_value"]),
                unit=c.get("unit"),
            )
            for c in entry.get("attribute_changes", [])
        ]
        scenarios.append(ImpactScenario(
            id=entry["id"],
            description=entry["description"],
            component_local=entry["component_local"],
            attribute_changes=changes,
            ground_truth_violations=entry.get("ground_truth_violations", []),
        ))

    return namespace, scenarios
