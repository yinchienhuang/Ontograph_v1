"""
synthesizer/self_check.py — Verify that the generated document faithfully
encodes all approved ontology triples.

Strategy:
  For every approved OntologyDeltaEntry whose triple.object is a *literal*
  (i.e., triple.datatype is set, OR the object is not an IRI):
    - Search the generated Markdown for the expected value.
    - For numeric literals, allow ±1% tolerance.
    - For string literals, require exact substring match.

  Object-property triples (object is an IRI) are not checked here because
  their "presence" in text is ambiguous — the label might appear in many forms.

After running, attach the result to the document and optionally log warnings.
"""

from __future__ import annotations

import re

from ontograph.models.ontology import OntologyDelta, OntologyDeltaEntry
from ontograph.models.synthesis import FactCheckItem, SelfCheckResult, SynthesizedDocument


# ---------------------------------------------------------------------------
# Value matching helpers
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _is_numeric(value: str) -> bool:
    """Return True if `value` is a plain number (int or float)."""
    return bool(_NUMERIC_RE.fullmatch(value.strip()))


def _extract_numeric(value: str) -> float | None:
    """Parse the first number found in `value`, e.g. '89 kg' → 89.0."""
    m = _NUMERIC_RE.search(value)
    return float(m.group()) if m else None


def _is_object_iri(object_value: str, datatype: str | None) -> bool:
    """
    Heuristic: is this triple object an IRI rather than a literal?
    Object-property triples have no datatype and the object looks like an IRI.
    """
    if datatype:
        return False
    return object_value.startswith("http") or (
        ":" in object_value and not object_value.startswith('"')
    )


def _values_match(expected: str, found: str | None) -> tuple[bool, str | None]:
    """
    Compare expected ontology value to found text value.

    Returns (match: bool, note: str | None).
    """
    if found is None:
        return False, "value not found in generated text"

    # Numeric comparison with 1% tolerance
    exp_num = _extract_numeric(expected)
    found_num = _extract_numeric(found)
    if exp_num is not None and found_num is not None:
        if exp_num == 0.0:
            match = found_num == 0.0
        else:
            match = abs(exp_num - found_num) / abs(exp_num) <= 0.01
        if not match:
            return False, f"numeric mismatch: expected ~{exp_num}, found {found_num}"
        return True, None

    # Exact string match (case-insensitive)
    if expected.lower() in found.lower():
        return True, None

    return False, f"string mismatch: expected '{expected}', found '{found}'"


def _search_value_in_text(expected: str, markdown: str) -> str | None:
    """
    Search for `expected` (or its numeric portion) anywhere in `markdown`.

    Returns the matched substring, or None if not found.

    Strategy:
      1. Try exact substring match first.
      2. If expected contains a number, search for that number in the text
         (it may appear with a different unit suffix or formatting).
    """
    # 1. Exact substring (case-insensitive)
    if expected.lower() in markdown.lower():
        return expected

    # 2. Numeric search
    exp_num = _extract_numeric(expected)
    if exp_num is not None:
        # Find all numeric tokens in the markdown and return the closest match
        for m in _NUMERIC_RE.finditer(markdown):
            candidate = float(m.group())
            if exp_num == 0.0:
                if candidate == 0.0:
                    return m.group()
            elif abs(exp_num - candidate) / abs(exp_num) <= 0.01:
                return m.group()

    return None


# ---------------------------------------------------------------------------
# Entry filter: which triples are checkable?
# ---------------------------------------------------------------------------

def _is_checkable(entry: OntologyDeltaEntry) -> bool:
    """
    Return True for triples whose object value can be looked up in generated text.

    We check:
      - Triples with an explicit XSD datatype (guaranteed literal)
      - Triples with a plain-string object that is NOT an IRI
    We skip:
      - Object-property triples (object is an IRI → label resolution is ambiguous)
      - rdf:type triples (type assignment is hard to verify in free text)
    """
    triple = entry.triple

    # Skip rdf:type
    local_pred = triple.predicate.split("#")[-1].split(":")[-1].split("/")[-1]
    if local_pred == "type":
        return False

    # Has an XSD datatype → definitely a checkable literal
    if triple.datatype:
        return True

    # No datatype: check if object looks like an IRI
    if _is_object_iri(triple.object, triple.datatype):
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_self_check(
    doc: SynthesizedDocument,
    delta: OntologyDelta,
) -> SelfCheckResult:
    """
    Re-verify the generated document against its source ontology triples.

    For each checkable approved entry, searches for the expected value in
    `doc.markdown` and records whether it was found and matched.

    Returns a SelfCheckResult. Does NOT mutate `doc` — call
    `attach_self_check()` to persist the result on the document object.
    """
    approved = delta.approved_entries()
    checkable = [e for e in approved if _is_checkable(e)]

    items: list[FactCheckItem] = []
    for entry in checkable:
        found = _search_value_in_text(entry.triple.object, doc.markdown)
        match, note = _values_match(entry.triple.object, found)
        items.append(FactCheckItem(
            triple_id=entry.id,
            expected_object=entry.triple.object,
            found_text=found,
            match=match,
            note=note,
        ))

    matched = sum(1 for i in items if i.match)
    total = len(items)

    return SelfCheckResult(
        checked_triple_count=total,
        matched_count=matched,
        coverage=matched / total if total > 0 else 1.0,
        items=items,
    )


def attach_self_check(
    doc: SynthesizedDocument,
    delta: OntologyDelta,
) -> SynthesizedDocument:
    """
    Run self-check and return a new SynthesizedDocument with the result attached.

    The original `doc` is not mutated (Pydantic model_copy is used).
    """
    result = run_self_check(doc, delta)
    return doc.model_copy(update={"self_check": result})


def format_self_check_report(result: SelfCheckResult) -> str:
    """
    Format a SelfCheckResult as a human-readable text summary.

    Useful for CLI output and logging.
    """
    lines: list[str] = [
        f"Self-check: {result.matched_count}/{result.checked_triple_count} "
        f"triples verified  (coverage: {result.coverage:.0%})",
    ]
    if result.discrepancies:
        lines.append(f"\n{len(result.discrepancies)} discrepancy(ies):")
        for item in result.discrepancies:
            lines.append(
                f"  [{item.triple_id}]  expected: '{item.expected_object}'  "
                f"found: {item.found_text!r}  — {item.note or ''}"
            )
    else:
        lines.append("  All values verified successfully.")
    return "\n".join(lines)
