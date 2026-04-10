"""
rules/schema.py — Pydantic models for structured compatibility rules and violations.

Two concepts:
  OrgRule         — A structured threshold rule authored in YAML.
  ViolationReport — The result of running check_rules() against OWL or document data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

class RuleWhen(BaseModel):
    """Threshold condition that triggers a rule violation."""

    attribute: str           # property local name, e.g. "hasDryMass"
    operator:  str           # ">", "<", ">=", "<=", "==", "!="
    value:     float | int | str
    unit:      str | None = None


class OrgRule(BaseModel):
    """
    One structured compatibility/constraint rule.

    Defined in a YAML rules file and used by the checker to detect violations
    in a working OWL ontology (ontology mode) or synthesized document text
    (document mode).
    """

    id:                str
    name:              str
    namespace:         str = ""            # resolved from YAML top-level namespace field
    subject_type:      str | None = None   # OWL class local name; None = any subject
    object_type:       str | None = None   # None = single-entity rule (no object pairing)
    when:              RuleWhen
    consequence:       str = ""
    severity:          Literal["critical", "warning", "info"] = "warning"
    note:              str = ""
    plain_english:     str = ""            # auto-generated vague description if blank
    expected_violated: bool = False        # deterministic ground truth — set in YAML


# ---------------------------------------------------------------------------
# Violation result
# ---------------------------------------------------------------------------

class ViolationInstance(BaseModel):
    """
    One evaluated (subject, object) pair for a rule.

    Both violated and non-violated pairs may appear — use ViolationReport.critical()
    to filter for actual violations.
    """

    rule_id:       str
    severity:      Literal["critical", "warning", "info"] = "warning"
    mode:          Literal["ontology", "document"]
    subject_label: str
    object_label:  str | None = None
    violated:      bool
    confidence:    float = Field(ge=0.0, le=1.0)
    explanation:   str
    source_refs:   list[str] = Field(
        default_factory=list,
        description=(
            "Ontology mode: extracted OWL triple(s), e.g. 'BatteryPack1 · operatingTempMinC = -10 °C'. "
            "Document mode: verbatim quoted sentence(s) from the synthesized document."
        ),
    )


class ModeTokenUsage(BaseModel):
    """Token counts for one checking mode (ontology or document)."""
    input_tokens:  int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class ViolationReport(BaseModel):
    """
    Full violation check report produced by check_rules().

    Contains results from ontology mode, document mode, or both.
    Saved to data/violations/ as JSON.
    """

    id:            str = Field(description="sha256(rules_file + timestamp)[:16]")
    created_at:    str = Field(description="ISO 8601 UTC timestamp")
    rules_file:    str
    owl_path:      str | None = None
    document_path: str | None = None
    mode:          str
    violations:    list[ViolationInstance]
    ontology_tokens: ModeTokenUsage = Field(default_factory=ModeTokenUsage)
    document_tokens: ModeTokenUsage = Field(default_factory=ModeTokenUsage)

    def critical(self) -> list[ViolationInstance]:
        """Return only instances where violated=True (actual rule violations)."""
        return [v for v in self.violations if v.violated]

    def by_rule(self, rule_id: str) -> list[ViolationInstance]:
        """Return all ViolationInstances for a specific rule_id."""
        return [v for v in self.violations if v.rule_id == rule_id]


# ---------------------------------------------------------------------------
# Conflict scoring models
# ---------------------------------------------------------------------------

class RuleScore(BaseModel):
    """Scoring outcome for one rule against one detection arm."""

    rule_id:                str
    rule_name:              str
    severity:               Literal["critical", "warning", "info"]
    gt_violated:            bool
    detected:               bool
    match_score:            float = 0.0
    matched_conflict_index: int | None = None
    outcome:                Literal["TP", "FP", "FN", "TN"]


class ConflictScoreReport(BaseModel):
    """P/R/F1 evaluation of both conflict-detection arms against ground truth."""

    model_config = {"protected_namespaces": ()}

    ontograph_precision: float
    ontograph_recall:    float
    ontograph_f1:        float
    direct_precision:    float
    direct_recall:       float
    direct_f1:           float
    ontograph_tokens:    dict[str, int]   # {"input": N, "output": M}
    direct_tokens:       dict[str, int]
    per_rule_ontograph:  list[RuleScore]
    per_rule_direct:     list[RuleScore]
