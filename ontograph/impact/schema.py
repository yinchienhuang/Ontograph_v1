"""
impact/schema.py — Pydantic models for design-change impact analysis.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AttributeChangeSpec(BaseModel):
    """One attribute modification applied to a component individual."""

    property_local: str       # OWL property local name, e.g. "powerW"
    old_value:      str       # Original value (string form)
    new_value:      str       # Replacement value (string form)
    unit:           str | None = None


class ImpactScenario(BaseModel):
    """
    One design-change scenario to evaluate.

    ``component_local``         — OWL individual being modified (local name, e.g. "OBC1").
    ``attribute_changes``       — Property values to update in the working OWL.
    ``ground_truth_violations`` — Rule IDs that SHOULD be violated AFTER the change.
                                  Provided externally (e.g. from an OWL reasoner).
    """

    id:                      str
    description:             str
    component_local:         str
    attribute_changes:       list[AttributeChangeSpec]
    ground_truth_violations: list[str] = Field(
        default_factory=list,
        description="Rule IDs that should be violated AFTER the change (from external reasoner).",
    )


class ImpactArmResult(BaseModel):
    """Per-arm (ontology or document) scoring result for one scenario."""

    arm:              Literal["ontology", "document"]
    violations_after: list[str]   # rule IDs the arm detected as violated post-change
    precision:        float
    recall:           float
    f1:               float


class ImpactAnalysisResult(BaseModel):
    """
    Full result for one impact analysis scenario.

    Produced by analyze_impact() and saved to data/evaluations/.
    """

    model_config = {"protected_namespaces": ()}

    id:                      str = Field(description="sha256(scenario_id|owl_path|timestamp)[:16]")
    created_at:              str
    scenario_id:             str
    description:             str
    scenarios_file:          str
    owl_path:                str
    document_path:           str | None
    baseline_violations:     list[str]   # rule IDs violated BEFORE the change
    ground_truth_violations: list[str]   # rule IDs that should be violated AFTER
    arms:                    list[ImpactArmResult]
    winner:                  str | None  # arm name with highest F1; None if tied or single arm

    def arm(self, name: str) -> ImpactArmResult | None:
        """Return the arm result for the given arm name, or None if not present."""
        return next((a for a in self.arms if a.arm == name), None)
