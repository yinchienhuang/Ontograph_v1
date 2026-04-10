"""
rules/conflict_detector.py — Zero-rule autonomous conflict detection.

Two arms compare how well each method finds cross-entity design conflicts
without any pre-written rules:

  ontograph arm
      Step 1 — Extract all individuals and their exact property values from
               the working OWL graph.
      Step 2 — LLM generates candidate cross-entity constraint rules from
               the structured data.
      Step 3 — LLM checks each generated rule against the exact OWL values.
      Result: conflicts grounded in precise, programmatically extracted data.

  direct arm
      Step 1 — Feed the full raw document text to the LLM.
      Step 2 — LLM identifies conflicts in a single pass from prose alone.
      Result: conflicts inferred from vague, potentially incomplete text.

Public API:
    detect_conflicts(owl_path, document_path, provider, mode) -> ConflictReport
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rdflib import BNode, Graph, URIRef
from rdflib.namespace import RDF

from ontograph.llm import LLMMessage, LLMProvider, LLMRequest
from ontograph.utils.owl import _SCHEMA_TYPES, load_graph


# ---------------------------------------------------------------------------
# Shared output schema
# ---------------------------------------------------------------------------

class ConflictInstance(BaseModel):
    """One detected cross-entity conflict."""
    subject:       str                                  # first entity name
    object:        str | None = None                    # second entity (None = single-entity)
    conflict_type: str                                  # short label, e.g. "power_budget_exceeded"
    description:   str                                  # human-readable explanation
    confidence:    float = Field(ge=0.0, le=1.0)
    severity:      Literal["critical", "warning", "info"] = "warning"
    evidence:      list[str] = Field(
        default_factory=list,
        description=(
            "Ontograph arm: OWL triple strings that support this conflict. "
            "Direct arm: verbatim sentence(s) quoted from the document."
        ),
    )


class ConflictReport(BaseModel):
    """Full conflict detection report from detect_conflicts()."""
    model_config = {"protected_namespaces": ()}

    id:                  str
    created_at:          str
    mode:                Literal["ontograph", "direct", "both"]
    owl_path:            str | None = None
    document_path:       str | None = None
    provider_id:         str = ""
    ontograph_conflicts: list[ConflictInstance] = Field(default_factory=list)
    direct_conflicts:    list[ConflictInstance] = Field(default_factory=list)
    ontograph_tokens:    dict[str, int] = Field(
        default_factory=lambda: {"input": 0, "output": 0},
        description="Token usage for the ontograph arm (step1 + step2 + step3).",
    )
    direct_tokens:       dict[str, int] = Field(
        default_factory=lambda: {"input": 0, "output": 0},
        description="Token usage for the direct arm.",
    )


# ---------------------------------------------------------------------------
# OWL helpers — extract all ABox individuals with their properties
# ---------------------------------------------------------------------------

def _extract_all_individuals(g: Graph, namespace: str) -> dict[str, dict[str, str]]:
    """
    Return {local_name: {pred_local: value_str}} for every ABox individual.

    Multi-valued properties are joined with "; ".
    Object property values are resolved to their local name when they are
    within the same namespace; otherwise kept as the full IRI string.
    """
    schema_subjects: set[URIRef] = set()
    for schema_type in _SCHEMA_TYPES:
        for s in g.subjects(RDF.type, schema_type):
            if not isinstance(s, BNode):
                schema_subjects.add(s)  # type: ignore[arg-type]

    # Collect ABox individuals
    abox_iris: set[str] = set()
    for s in g.subjects(RDF.type, None):
        if isinstance(s, BNode):
            continue
        if s in schema_subjects:
            continue
        abox_iris.add(str(s))

    result: dict[str, dict[str, list[str]]] = {}

    for iri in abox_iris:
        if not iri.startswith(namespace):
            continue
        local = iri[len(namespace):]
        props: dict[str, list[str]] = defaultdict(list)
        uri = URIRef(iri)
        for _s, p, o in g.triples((uri, None, None)):
            if isinstance(o, BNode):
                continue
            pred_local = str(p).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            if pred_local == "type":
                obj_str = str(o).replace(namespace, "")
                props["rdf_type"].append(obj_str)
            else:
                obj_str = str(o).replace(namespace, "")
                props[pred_local].append(obj_str)
        result[local] = props

    return {
        local: {k: (v[0] if len(v) == 1 else "; ".join(v)) for k, v in props.items()}
        for local, props in result.items()
    }


def _detect_namespace(g: Graph) -> str:
    """Return the most common non-OWL/RDF namespace in the graph."""
    skip = {"http://www.w3.org/", "http://www.w3.org/2002/", "http://www.w3.org/2000/",
            "http://www.w3.org/1999/", "http://purl.org/"}
    counts: dict[str, int] = defaultdict(int)
    for s in g.subjects():
        s_str = str(s)
        if "#" in s_str:
            ns = s_str.rsplit("#", 1)[0] + "#"
        elif "/" in s_str:
            ns = s_str.rsplit("/", 1)[0] + "/"
        else:
            continue
        if any(s_str.startswith(sk) for sk in skip):
            continue
        counts[ns] += 1
    return max(counts, key=counts.get) if counts else ""


def _format_entity_block(individuals: dict[str, dict[str, str]]) -> str:
    """Format all individuals into a compact block for LLM prompts."""
    lines = []
    for name, props in sorted(individuals.items()):
        lines.append(f"  {name}:")
        for pred, val in sorted(props.items()):
            lines.append(f"    {pred}: {val}")
    return "\n".join(lines) if lines else "  (none found)"


# ---------------------------------------------------------------------------
# Exact-hint formatter (used by scripts/detect_conflicts.py for --hint-mode exact)
# ---------------------------------------------------------------------------

def _format_exact_hint(rule: object) -> str:
    """
    Format one OrgRule as a precise specification string (threshold included).
    Imported by scripts/detect_conflicts.py — avoids importing schema in this module.
    """
    unit = (f" {rule.when.unit}" if rule.when.unit else "")  # type: ignore[attr-defined]
    obj_or_subj = rule.object_type or rule.subject_type or "entity"  # type: ignore[attr-defined]
    lines = [
        f"[{rule.id}] {rule.name} ({rule.severity})",  # type: ignore[attr-defined]
        f"  Check: {rule.when.attribute} of {obj_or_subj} "  # type: ignore[attr-defined]
        f"{rule.when.operator} {rule.when.value}{unit}",  # type: ignore[attr-defined]
    ]
    note = (rule.note or "").strip()  # type: ignore[attr-defined]
    if note:
        lines.append(f"  Note: {note[:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM schemas — ontograph arm (step 2: rule discovery)
# ---------------------------------------------------------------------------

class DiscoveredRule(BaseModel):
    """One cross-entity constraint discovered by the LLM from OWL data."""
    subject_entity:  str   = Field(description="Local name of the first entity")
    object_entity:   str   = Field(description="Local name of the second entity (or same as subject for single-entity)")
    attribute:       str   = Field(description="The property that was checked")
    operator:        str   = Field(description=">  <  >=  <=  ==  !=")
    threshold:       float = Field(description="Numeric threshold derived from the other entity's value")
    threshold_basis: str   = Field(description="How the threshold was derived, e.g. '1.5 × OBC1.powerW = 3.0'")
    rationale:       str   = Field(description="Why this constraint should hold (1-2 sentences)")
    severity:        Literal["critical", "warning", "info"] = "warning"


class DiscoveredRuleSet(BaseModel):
    rules: list[DiscoveredRule]


# ---------------------------------------------------------------------------
# LLM schemas — ontograph arm (step 3: rule checking)
# ---------------------------------------------------------------------------

class RuleViolationCheck(BaseModel):
    violated:    bool
    confidence:  float = Field(ge=0.0, le=1.0)
    explanation: str
    evidence:    list[str] = Field(
        default_factory=list,
        description="OWL triple strings that directly support this verdict, e.g. 'Radio1 · powerW = 4.0 W'",
    )


# ---------------------------------------------------------------------------
# LLM schemas — direct arm
# ---------------------------------------------------------------------------

class DirectConflictInstance(BaseModel):
    subject:       str
    object:        str | None = None
    conflict_type: str
    description:   str
    confidence:    float = Field(ge=0.0, le=1.0)
    severity:      Literal["critical", "warning", "info"] = "warning"
    evidence:      list[str] = Field(
        default_factory=list,
        description="1-2 sentences quoted VERBATIM from the document.",
    )


class DirectConflictResponse(BaseModel):
    conflicts: list[DirectConflictInstance]


# ---------------------------------------------------------------------------
# Ontograph arm — step 2: rule discovery from OWL individuals
# ---------------------------------------------------------------------------

_RULE_DISCOVERY_SYSTEM = (
    "You are an aerospace systems engineering expert. "
    "Given a list of design entities and their exact property values extracted from an OWL ontology, "
    "identify cross-entity compatibility constraints that should hold between pairs of entities. "
    "Focus on constraints where the threshold for one entity is DERIVED from another entity's value "
    "(e.g., radio power must not exceed 1.5x OBC power; battery voltage must exceed 1.5x bus voltage). "
    "Do NOT invent constraints that cannot be grounded in the given property values. "
    "Each rule must reference real property names and real entity names from the input. "
    "Produce between 3 and 8 rules. Prefer critical and warning severities."
)


def _discover_rules(
    individuals: dict[str, dict[str, str]],
    provider: LLMProvider,
    ontograph_hints: list[str] | None = None,
) -> tuple[list[DiscoveredRule], int, int]:
    """Step 2: Ask LLM to discover cross-entity constraints from OWL data."""
    entity_block = _format_entity_block(individuals)

    hints_block = ""
    if ontograph_hints:
        hints_section = "\n".join(f"  - {h}" for h in ontograph_hints)
        hints_block = (
            "\n\nOrganizational compatibility guidelines:\n"
            + hints_section
            + "\n\nUse these guidelines to focus your rule discovery on the right "
              "entity pairs and concern areas."
        )

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=_RULE_DISCOVERY_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    "Here are all design entities and their exact property values:\n\n"
                    f"{entity_block}"
                    f"{hints_block}\n\n"
                    "Identify cross-entity compatibility constraints. "
                    "For each rule, derive the threshold from another entity's property value. "
                    "Return a list of discovered rules."
                ),
            ),
        ],
        response_model=DiscoveredRuleSet,
        temperature=0.1,
    )
    response = provider.complete(request)
    rules = response.parsed.rules
    return rules, response.usage.input_tokens, response.usage.output_tokens


# ---------------------------------------------------------------------------
# Ontograph arm — step 3: check each discovered rule against OWL values
# ---------------------------------------------------------------------------

_RULE_CHECK_SYSTEM = (
    "You are an ontology rule checker. "
    "Given a specific constraint and exact property values from an OWL ontology, "
    "determine whether the constraint is violated. "
    "Use only the values provided — do not infer or estimate. "
    "Populate evidence with OWL triple strings in the format 'EntityName · property = value'."
)


def _check_discovered_rule(
    rule: DiscoveredRule,
    individuals: dict[str, dict[str, str]],
    provider: LLMProvider,
) -> tuple[ConflictInstance | None, int, int]:
    """Step 3: Check one discovered rule against exact OWL values."""
    subj_props = individuals.get(rule.subject_entity, {})
    obj_props  = individuals.get(rule.object_entity, {})

    prop_block = (
        f"{rule.subject_entity}:\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(subj_props.items()))
        + (
            f"\n{rule.object_entity}:\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(obj_props.items()))
            if rule.object_entity != rule.subject_entity else ""
        )
    )

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=_RULE_CHECK_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    f"Constraint: {rule.attribute} of {rule.object_entity} "
                    f"{rule.operator} {rule.threshold} "
                    f"(basis: {rule.threshold_basis})\n"
                    f"Rationale: {rule.rationale}\n\n"
                    f"Entity values:\n{prop_block}\n\n"
                    "Is this constraint violated?"
                ),
            ),
        ],
        response_model=RuleViolationCheck,
    )
    response = provider.complete(request)
    check = response.parsed

    if not check.violated:
        return None, response.usage.input_tokens, response.usage.output_tokens

    conflict = ConflictInstance(
        subject=rule.subject_entity,
        object=rule.object_entity if rule.object_entity != rule.subject_entity else None,
        conflict_type=f"{rule.attribute}_{rule.operator.strip()}_threshold",
        description=f"{rule.rationale} ({rule.threshold_basis})",
        confidence=check.confidence,
        severity=rule.severity,
        evidence=check.evidence or [
            f"{rule.object_entity} · {rule.attribute} (threshold: {rule.operator} {rule.threshold})"
        ],
    )
    return conflict, response.usage.input_tokens, response.usage.output_tokens


# ---------------------------------------------------------------------------
# Ontograph arm — orchestrator
# ---------------------------------------------------------------------------

def _run_ontograph_arm(
    owl_path: Path,
    provider: LLMProvider,
    ontograph_hints: list[str] | None = None,
) -> tuple[list[ConflictInstance], dict[str, int]]:
    """
    Two-pass ontograph conflict detection:
      1. Load OWL → extract all individuals + properties
      2. LLM discovers candidate cross-entity rules
      3. LLM checks each rule against exact OWL values
    """
    g = load_graph(owl_path, fmt="xml")
    namespace = _detect_namespace(g)
    individuals = _extract_all_individuals(g, namespace)

    total_in = total_out = 0

    # Step 2 — rule discovery
    rules, inp, out = _discover_rules(individuals, provider, ontograph_hints=ontograph_hints)
    total_in  += inp
    total_out += out

    # Step 3 — check each rule
    conflicts: list[ConflictInstance] = []
    for rule in rules:
        conflict, inp, out = _check_discovered_rule(rule, individuals, provider)
        total_in  += inp
        total_out += out
        if conflict is not None:
            conflicts.append(conflict)

    return conflicts, {"input": total_in, "output": total_out}


# ---------------------------------------------------------------------------
# Direct arm — single-pass document conflict detection
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM = (
    "You are a systems engineering conflict detection expert. "
    "Read the engineering document and identify cross-entity design conflicts or "
    "incompatibilities — cases where two or more components have property values "
    "that violate a design constraint or compatibility rule. "
    "Focus on quantitative conflicts (mass budgets, power limits, voltage ratios, "
    "thermal limits, etc.) where the document provides actual numbers. "
    "For each conflict, quote the verbatim sentence(s) from the document that support it. "
    "Do not report vague concerns — only conflicts supported by numbers in the document. "
    "If no clear conflicts are found, return an empty list."
)


def _run_direct_arm(
    document_path: Path,
    provider: LLMProvider,
    direct_hints: list[str] | None = None,
) -> tuple[list[ConflictInstance], dict[str, int]]:
    """Single-pass direct conflict detection from document text."""
    document_text = document_path.read_text(encoding="utf-8")

    hints_block = ""
    if direct_hints:
        hints_section = "\n".join(f"  - {h}" for h in direct_hints)
        hints_block = (
            "\n\nOrganizational compatibility guidelines:\n"
            + hints_section
            + "\n\nUse these guidelines to focus your conflict search."
        )

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=_DIRECT_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    f"Engineering document:\n\n{document_text}"
                    f"{hints_block}\n\n"
                    "Identify all cross-entity design conflicts supported by numbers in the document."
                ),
            ),
        ],
        response_model=DirectConflictResponse,
        temperature=0.0,
    )
    response = provider.complete(request)

    conflicts = [
        ConflictInstance(
            subject=c.subject,
            object=c.object,
            conflict_type=c.conflict_type,
            description=c.description,
            confidence=c.confidence,
            severity=c.severity,
            evidence=c.evidence,
        )
        for c in response.parsed.conflicts
    ]
    tokens = {"input": response.usage.input_tokens, "output": response.usage.output_tokens}
    return conflicts, tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_conflicts(
    provider:         LLMProvider,
    owl_path:         Path | str | None = None,
    document_path:    Path | str | None = None,
    mode:             Literal["ontograph", "direct", "both"] = "both",
    ontograph_hints:  list[str] | None = None,
    direct_hints:     list[str] | None = None,
) -> ConflictReport:
    """
    Detect cross-entity design conflicts autonomously (no pre-written rules).

    Args:
        provider:      LLM provider instance.
        owl_path:      Path to working OWL file (required for ontograph/both mode).
        document_path: Path to synthesized document (required for direct/both mode).
        mode:          Which arm(s) to run.

    Returns:
        :class:`ConflictReport` with conflicts from each arm and token usage.
    """
    needs_owl = mode in ("ontograph", "both")
    needs_doc = mode in ("direct", "both")

    if needs_owl and owl_path is None:
        raise ValueError(f"mode='{mode}' requires owl_path")
    if needs_doc and document_path is None:
        raise ValueError(f"mode='{mode}' requires document_path")

    owl_path      = Path(owl_path)      if owl_path      else None
    document_path = Path(document_path) if document_path else None

    ont_conflicts: list[ConflictInstance] = []
    dir_conflicts: list[ConflictInstance] = []
    ont_tokens = {"input": 0, "output": 0}
    dir_tokens = {"input": 0, "output": 0}

    if needs_owl and owl_path:
        ont_conflicts, ont_tokens = _run_ontograph_arm(owl_path, provider, ontograph_hints=ontograph_hints)

    if needs_doc and document_path:
        dir_conflicts, dir_tokens = _run_direct_arm(document_path, provider, direct_hints=direct_hints)

    report_id = hashlib.sha256(
        f"{owl_path}|{document_path}|{mode}|{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:16]

    return ConflictReport(
        id=report_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        owl_path=str(owl_path) if owl_path else None,
        document_path=str(document_path) if document_path else None,
        ontograph_conflicts=ont_conflicts,
        direct_conflicts=dir_conflicts,
        ontograph_tokens=ont_tokens,
        direct_tokens=dir_tokens,
    )
