"""
ingest/org_loader.py — Load organizational knowledge rules from a YAML file.

Converts human-authored YAML rules into OntologyDeltaEntry objects tagged
ChangeSource.MANUAL so the evaluator's Arm B can use them alongside
pipeline-extracted triples.

YAML format
-----------
    namespace: "http://example.org/aerospace#"   # default IRI prefix

    predicates:                                   # optional shorthand aliases
        notCompatibleWith: "http://example.org/aerospace#notCompatibleWith"

    rules:
        - id: "org-001"
          subject:   "ComponentA"      # short name → expanded with namespace
          predicate: "notCompatibleWith"
          object:    "ComponentB"
          note:      "human-readable explanation stored as rationale"
          # optional:
          datatype:  "http://www.w3.org/2001/XMLSchema#float"  # for literal objects

Short names (no "http://" prefix, no colon) are expanded using `namespace`.
Alias names (defined in `predicates:`) are resolved to their full IRIs.
Values that already start with "http://" or "https://" are used unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ontograph.models.ontology import ChangeSource, OntologyDeltaEntry, OntologyTriple


# ---------------------------------------------------------------------------
# IRI resolution helper
# ---------------------------------------------------------------------------

def _resolve_iri(value: str, namespace: str, aliases: dict[str, str]) -> str:
    """Expand a short name, alias, or full IRI to a canonical full IRI string.

    Resolution priority (first match wins):
      1. Already a full IRI (starts with http:// or https://)
      2. Defined in the ``aliases`` map
      3. Contains a colon (e.g. xsd:float) — passed through unchanged
      4. Short name: prepend ``namespace``
    """
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value in aliases:
        return aliases[value]
    if ":" in value:
        return value          # prefixed notation, leave as-is
    return namespace + value  # bare local name → expand with default namespace


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_org_knowledge(path: str | Path) -> list[OntologyDeltaEntry]:
    """Parse a YAML org-knowledge file and return MANUAL OntologyDeltaEntries.

    All returned entries have:
      - ``change_source = ChangeSource.MANUAL``
      - ``status = "approved"``
      - ``confidence = 1.0``
      - no ``source_chunk_id`` or ``source_entity_id`` (human-authored)

    Args:
        path: Path to a YAML file following the org-knowledge schema.

    Returns:
        List of :class:`~ontograph.models.ontology.OntologyDeltaEntry` objects,
        one per rule, ready to be added to the working OWL graph.

    Raises:
        KeyError: If a rule is missing a required field (``id``, ``subject``,
            ``predicate``, or ``object``).
        yaml.YAMLError: If the file is not valid YAML.
    """
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return []

    data: dict[str, Any] = raw
    namespace: str = str(data.get("namespace", ""))
    aliases: dict[str, str] = {
        str(k): str(v) for k, v in (data.get("predicates") or {}).items()
    }
    rules: list[dict[str, Any]] = data.get("rules") or []

    entries: list[OntologyDeltaEntry] = []
    for rule in rules:
        rule_id    = str(rule["id"])
        subject    = _resolve_iri(str(rule["subject"]),   namespace, aliases)
        predicate  = _resolve_iri(str(rule["predicate"]), namespace, aliases)
        obj_raw    = str(rule["object"])
        datatype   = rule.get("datatype")
        note       = str(rule.get("note") or "").strip()

        # Objects with an explicit datatype are literals; others are IRIs
        if datatype:
            object_str = obj_raw
        else:
            object_str = _resolve_iri(obj_raw, namespace, aliases)

        triple = OntologyTriple(
            subject=subject,
            predicate=predicate,
            object=object_str,
            datatype=datatype or None,
        )
        entry = OntologyDeltaEntry(
            id=rule_id,
            triple=triple,
            rationale=note,
            confidence=1.0,
            change_source=ChangeSource.MANUAL,
            status="approved",
        )
        entries.append(entry)

    return entries
