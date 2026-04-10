"""
ontograph/rules/conflict_scorer.py — Ground-truth P/R/F1 scoring for conflict detection.

Matches each ConflictInstance from an arm against the deterministic ground truth
stored in rule.expected_violated (set in the YAML rules file).

Matching strategy (two signals, combined):
  1. Attribute match — ci.conflict_type contains rule.when.attribute.
     Reliable for the ontograph arm, which generates conflict_type programmatically.
  2. Normalized name similarity — instance names are stripped of trailing digits /
     single-letter suffixes before CamelCase Jaccard + prefix comparison against
     the rule's subject_type / object_type class names.

Greedy one-to-one assignment prevents double-counting.

Public API
----------
    score_conflicts(report, rules) -> ConflictScoreReport
"""

from __future__ import annotations

import re

from ontograph.rules.conflict_detector import ConflictInstance, ConflictReport
from ontograph.rules.schema import ConflictScoreReport, OrgRule, RuleScore
from ontograph.utils.iri_align import iri_similarity

MATCH_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_instance(name: str) -> str:
    """
    Strip trailing digits then trailing single uppercase letter (if preceded by
    lowercase) from instance names so they compare better against class names.

    Examples:
        Radio1   → Radio       (strip digit)
        PDU1     → PDU         (strip digit, keep acronym)
        EngineA  → Engine      (strip variant letter)
        RouteA   → Route
        JobyS4   → Joby        (strip digit, then trailing uppercase S after y)
        Tank2    → Tank
    """
    s = re.sub(r'\d+$', '', name)                  # strip trailing digits
    s = re.sub(r'(?<=[a-z])[A-Z]$', '', s)         # strip trailing variant letter
    return s or name


def _class_instance_sim(class_name: str, instance_name: str) -> float:
    """
    Compare one instance name against one class name using three strategies:
      - raw iri_similarity
      - iri_similarity after normalizing the instance name
      - prefix check (class starts with norm, or norm starts with class)
    Returns the max score.
    """
    if not class_name or not instance_name:
        return 0.0
    norm = _normalize_instance(instance_name)
    raw    = iri_similarity(instance_name, class_name)
    normed = iri_similarity(norm, class_name)
    prefix = 1.0 if (
        class_name.lower().startswith(norm.lower()) or
        norm.lower().startswith(class_name.lower())
    ) else 0.0
    return max(raw, normed, prefix)


# ---------------------------------------------------------------------------
# Combined conflict ↔ rule similarity
# ---------------------------------------------------------------------------

def _conflict_rule_sim(ci: ConflictInstance, rule: OrgRule) -> float:
    """
    Combined similarity between a detected conflict and a rule.

    Signal 1 — attribute match:
        If ci.conflict_type contains rule.when.attribute, this is a strong match.
        The ontograph arm generates conflict_type = "{attribute}_{op}_threshold".

    Signal 2 — normalized name similarity:
        Average of subject and object class-instance similarities, trying both
        orderings (forward and swapped).

    Combined score:
        attr_match  → base 0.5; +0.5 * name_score
        no match    → 0.6 * name_score  (pure name, slightly discounted)
    """
    attr_match = bool(rule.when.attribute) and rule.when.attribute in (ci.conflict_type or "")

    subj_cls = rule.subject_type or ""
    obj_cls  = rule.object_type  or ""
    ci_subj  = ci.subject or ""
    ci_obj   = ci.object  or ""

    forward = (_class_instance_sim(subj_cls, ci_subj) +
               _class_instance_sim(obj_cls,  ci_obj)) / 2.0
    swapped = (_class_instance_sim(subj_cls, ci_obj) +
               _class_instance_sim(obj_cls,  ci_subj)) / 2.0
    name_score = max(forward, swapped)

    if attr_match:
        return 0.5 + 0.5 * name_score
    return 0.6 * name_score


# ---------------------------------------------------------------------------
# Per-arm scoring
# ---------------------------------------------------------------------------

def _score_arm(
    conflicts: list[ConflictInstance] | None,
    rules:     list[OrgRule],
) -> list[RuleScore]:
    """
    Score one detection arm against deterministic ground truth.

    Ground truth comes from rule.expected_violated — no LLM involved.
    Uses greedy one-to-one assignment (highest similarity first).
    """
    if conflicts is None:
        conflicts = []

    violated_rules  = [r for r in rules if r.expected_violated]
    compliant_rules = [r for r in rules if not r.expected_violated]

    # ── TP candidates (violated rules) ───────────────────────────────────────
    candidates: list[tuple[float, int, str]] = []
    for rule in violated_rules:
        for ci_idx, ci in enumerate(conflicts):
            score = _conflict_rule_sim(ci, rule)
            if score >= MATCH_THRESHOLD:
                candidates.append((score, ci_idx, rule.id))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_ci:   set[int] = set()
    used_rule: set[str] = set()
    matched:   dict[str, tuple[int, float]] = {}

    for score, ci_idx, rule_id in candidates:
        if ci_idx in used_ci or rule_id in used_rule:
            continue
        matched[rule_id] = (ci_idx, score)
        used_ci.add(ci_idx)
        used_rule.add(rule_id)

    # ── FP candidates (compliant rules) ──────────────────────────────────────
    unmatched_ci = set(range(len(conflicts))) - used_ci

    fp_candidates: list[tuple[float, int, str]] = []
    for rule in compliant_rules:
        for ci_idx in unmatched_ci:
            ci = conflicts[ci_idx]
            score = _conflict_rule_sim(ci, rule)
            if score >= MATCH_THRESHOLD:
                fp_candidates.append((score, ci_idx, rule.id))

    fp_candidates.sort(key=lambda x: x[0], reverse=True)
    used_ci_fp:   set[int] = set()
    used_rule_fp: set[str] = set()
    fp_matched:   dict[str, tuple[int, float]] = {}

    for score, ci_idx, rule_id in fp_candidates:
        if ci_idx in used_ci_fp or rule_id in used_rule_fp:
            continue
        fp_matched[rule_id] = (ci_idx, score)
        used_ci_fp.add(ci_idx)
        used_rule_fp.add(rule_id)

    # ── Assemble per-rule scores ──────────────────────────────────────────────
    scores: list[RuleScore] = []
    for rule in rules:
        gt_violated = rule.expected_violated

        if gt_violated:
            if rule.id in matched:
                ci_idx, score = matched[rule.id]
                outcome      = "TP"
                detected     = True
                match_score  = score
                matched_idx: int | None = ci_idx
            else:
                outcome     = "FN"
                detected    = False
                match_score = 0.0
                matched_idx = None
        else:
            if rule.id in fp_matched:
                ci_idx, score = fp_matched[rule.id]
                outcome     = "FP"
                detected    = True
                match_score = score
                matched_idx = ci_idx
            else:
                outcome     = "TN"
                detected    = False
                match_score = 0.0
                matched_idx = None

        scores.append(RuleScore(
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            gt_violated=gt_violated,
            detected=detected,
            match_score=match_score,
            matched_conflict_index=matched_idx,
            outcome=outcome,  # type: ignore[arg-type]
        ))

    return scores


# ---------------------------------------------------------------------------
# P/R/F1 computation
# ---------------------------------------------------------------------------

def _compute_prf(scores: list[RuleScore]) -> tuple[float, float, float]:
    tp = sum(1 for s in scores if s.outcome == "TP")
    fp = sum(1 for s in scores if s.outcome == "FP")
    fn = sum(1 for s in scores if s.outcome == "FN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_conflicts(
    report: ConflictReport,
    rules:  list[OrgRule],
) -> ConflictScoreReport:
    """
    Score both detection arms against deterministic ground truth.

    Ground truth is read from rule.expected_violated — no LLM call needed.

    Parameters
    ----------
    report: ConflictReport from detect_conflicts()
    rules:  list[OrgRule] from load_rules() — must have expected_violated set

    Returns
    -------
    ConflictScoreReport with per-arm P/R/F1 and per-rule TP/FP/FN/TN breakdown.
    """
    ont_scores = _score_arm(report.ontograph_conflicts, rules)
    dir_scores = _score_arm(report.direct_conflicts,   rules)

    ont_p, ont_r, ont_f1 = _compute_prf(ont_scores)
    dir_p, dir_r, dir_f1 = _compute_prf(dir_scores)

    return ConflictScoreReport(
        ontograph_precision=ont_p,
        ontograph_recall=ont_r,
        ontograph_f1=ont_f1,
        direct_precision=dir_p,
        direct_recall=dir_r,
        direct_f1=dir_f1,
        ontograph_tokens=report.ontograph_tokens,
        direct_tokens=report.direct_tokens,
        per_rule_ontograph=ont_scores,
        per_rule_direct=dir_scores,
    )
