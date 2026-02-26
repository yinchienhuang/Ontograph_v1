"""
impact/analyzer.py — Core impact analysis logic.

Wraps check_rules() to compare rule violations before and after a design change.
Two arms are evaluated:

  ontology arm  — Checks rules on a *modified* OWL (precise, self-consistent after update).
  document arm  — Checks rules on the *original* synthesized document (stale after OWL change).

Arms are scored against user-supplied ground truth (from an external OWL reasoner).

Public API:
    analyze_impact(scenario, namespace, rules, rules_file, provider,
                   working_owl, document_path, mode) -> ImpactAnalysisResult
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from rdflib import Graph, Literal as RDFLiteral, URIRef
from rdflib.namespace import XSD

from ontograph.impact.schema import (
    ImpactArmResult,
    ImpactAnalysisResult,
    ImpactScenario,
)
from ontograph.llm import LLMProvider
from ontograph.rules.checker import check_rules
from ontograph.rules.schema import OrgRule
from ontograph.utils.owl import load_graph, save_graph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_change(owl_path: Path, namespace: str, scenario: ImpactScenario) -> Graph:
    """
    Return a modified copy of the OWL graph with the scenario's attribute changes applied.

    The original file is not modified — a new in-memory Graph is returned.
    Preserves the existing RDF literal datatype (e.g. xsd:decimal) for each
    modified triple so that rule checking continues to parse values correctly.
    """
    g = load_graph(owl_path, fmt="xml")

    for change in scenario.attribute_changes:
        subj = URIRef(namespace + scenario.component_local)
        pred = URIRef(namespace + change.property_local)

        # Detect existing literal datatype to preserve it
        existing = list(g.objects(subj, pred))
        dt = XSD.decimal
        if existing:
            first = existing[0]
            if hasattr(first, "datatype") and first.datatype is not None:
                dt = first.datatype

        g.remove((subj, pred, None))
        g.add((subj, pred, RDFLiteral(change.new_value, datatype=dt)))

    return g


def _score(
    predicted:    set[str],
    ground_truth: set[str],
) -> tuple[float, float, float]:
    """
    Return (precision, recall, F1) for set-based comparison.

    Comparison is case-insensitive.  Returns (0, 0, 0) when both sets are empty.
    """
    pred_lower = {r.lower() for r in predicted}
    gt_lower   = {r.lower() for r in ground_truth}

    tp        = len(pred_lower & gt_lower)
    precision = tp / len(pred_lower) if pred_lower else 0.0
    recall    = tp / len(gt_lower)   if gt_lower   else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1


def _build_arm_result(
    arm_name:     Literal["ontology", "document"],
    violated_set: set[str],
    ground_truth: set[str],
) -> ImpactArmResult:
    prec, rec, f1 = _score(violated_set, ground_truth)
    return ImpactArmResult(
        arm=arm_name,
        violations_after=sorted(violated_set),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_impact(
    scenario:      ImpactScenario,
    namespace:     str,
    rules:         list[OrgRule],
    rules_file:    str,
    provider:      LLMProvider,
    working_owl:   Path,
    document_path: Path | None = None,
    mode:          Literal["ontology", "document", "both"] = "both",
) -> ImpactAnalysisResult:
    """
    Run rule checking before and after a design change and score each arm.

    Steps:
      1. Baseline  — check_rules() on the original OWL.
      2. Modify    — apply scenario attribute_changes to a temporary OWL copy.
      3. Post-change — check_rules() on the modified OWL.
      4. Score     — compare each arm's post-change violations against ground truth.
      5. Return ImpactAnalysisResult.

    Args:
        scenario:      Design-change scenario (from load_scenarios).
        namespace:     OWL namespace IRI (e.g. "http://example.org/cubesat-ontology#").
        rules:         OrgRule list (plain_english must be populated for document/both mode).
        rules_file:    Path string for the rules YAML (stored in the report).
        provider:      LLM provider instance.
        working_owl:   Path to the working OWL file (never modified).
        document_path: Path to synthesized document (required for document/both mode).
        mode:          Which arm(s) to run.

    Returns:
        ImpactAnalysisResult with per-arm P/R/F1 scores.
    """
    # ── 1. Baseline check ────────────────────────────────────────────────────
    baseline = check_rules(
        rules=rules,
        provider=provider,
        working_owl=working_owl,
        document_path=document_path,
        mode=mode,
        rules_file=rules_file,
    )
    baseline_violated = {v.rule_id for v in baseline.critical()}

    # ── 2. Apply change → temp file ──────────────────────────────────────────
    modified_g = _apply_change(working_owl, namespace, scenario)

    with tempfile.NamedTemporaryFile(suffix=".owl", delete=False) as fh:
        temp_path = Path(fh.name)
    save_graph(modified_g, temp_path, fmt="xml")

    # ── 3. Post-change check ─────────────────────────────────────────────────
    try:
        post = check_rules(
            rules=rules,
            provider=provider,
            working_owl=temp_path,
            document_path=document_path,
            mode=mode,
            rules_file=rules_file,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    # ── 4. Split post violations by arm ──────────────────────────────────────
    ont_violated = {v.rule_id for v in post.violations if v.violated and v.mode == "ontology"}
    doc_violated = {v.rule_id for v in post.violations if v.violated and v.mode == "document"}
    gt = set(scenario.ground_truth_violations)

    # ── 5. Score arms ────────────────────────────────────────────────────────
    arms: list[ImpactArmResult] = []
    if mode in ("ontology", "both"):
        arms.append(_build_arm_result("ontology", ont_violated, gt))
    if mode in ("document", "both"):
        arms.append(_build_arm_result("document", doc_violated, gt))

    # ── 6. Determine winner ──────────────────────────────────────────────────
    winner: str | None = None
    if len(arms) == 1:
        winner = arms[0].arm
    elif len(arms) == 2:
        if arms[0].f1 > arms[1].f1:
            winner = arms[0].arm
        elif arms[1].f1 > arms[0].f1:
            winner = arms[1].arm
        # else tied → winner stays None

    # ── 7. Build result ──────────────────────────────────────────────────────
    report_id = hashlib.sha256(
        f"{scenario.id}|{working_owl}|{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:16]

    return ImpactAnalysisResult(
        id=report_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        scenario_id=scenario.id,
        description=scenario.description,
        scenarios_file=rules_file,
        owl_path=str(working_owl),
        document_path=str(document_path) if document_path else None,
        baseline_violations=sorted(baseline_violated),
        ground_truth_violations=sorted(gt),
        arms=arms,
        winner=winner,
    )
