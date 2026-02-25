"""
utils/owl_diff.py — Detect and reconcile manual edits to the working OWL file.

When a user edits the OWL file outside the pipeline (e.g. in Protégé), we
need to detect what changed so we can absorb those edits into the changelog
as MANUAL-sourced OntologyDeltaEntries — rather than silently overwriting them
on the next pipeline write.

Workflow:
    1. Before any pipeline write, call `diff_graphs(old_path, new_path)`.
    2. For any added triple not in our records, create a MANUAL entry.
    3. Log the changes in OntologyChangelog via `record_diff(...)`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ontograph.models.ontology import (
    ChangeSource,
    OntologyChangelog,
    OntologyChangelogEntry,
    OntologyDeltaEntry,
    OntologyTriple,
)
from ontograph.utils.io import sha256_file
from ontograph.utils.owl import graph_sha256, iter_triples, load_graph


# ---------------------------------------------------------------------------
# Diff result
# ---------------------------------------------------------------------------

@dataclass
class GraphDiff:
    """Triples added or removed between two OWL graph snapshots."""

    added: list[tuple[str, str, str]]    # (subject, predicate, object)
    removed: list[tuple[str, str, str]]

    @property
    def is_empty(self) -> bool:
        return not self.added and not self.removed


# ---------------------------------------------------------------------------
# Core diff
# ---------------------------------------------------------------------------

def diff_graphs(old_path: str | Path, new_path: str | Path) -> GraphDiff:
    """
    Compare two OWL files and return the triple-level difference.

    Blank nodes are skipped (they have no stable identity across serializations).
    """
    old_triples = set(iter_triples(load_graph(old_path)))
    new_triples = set(iter_triples(load_graph(new_path)))

    return GraphDiff(
        added=sorted(new_triples - old_triples),
        removed=sorted(old_triples - new_triples),
    )


def diff_from_snapshot(
    snapshot_path: str | Path,
    live_path: str | Path,
    known_entry_ids: set[str],
) -> list[OntologyDeltaEntry]:
    """
    Detect manually-added triples by diffing a snapshot against the live OWL.

    `snapshot_path`: OWL file as it was after the last pipeline write.
    `live_path`:     Current OWL file (may have been edited externally).
    `known_entry_ids`: Set of OntologyDeltaEntry.id already tracked.

    Returns new OntologyDeltaEntry objects (status=approved, source=MANUAL)
    for any triple that was added but not tracked.
    """
    diff = diff_graphs(snapshot_path, live_path)
    new_entries: list[OntologyDeltaEntry] = []

    for s, p, o in diff.added:
        entry_id = _make_entry_id(s, p, o)
        if entry_id in known_entry_ids:
            continue

        triple = OntologyTriple(subject=s, predicate=p, object=o)
        new_entries.append(
            OntologyDeltaEntry(
                id=entry_id,
                triple=triple,
                rationale="Detected as manual edit by owl_diff",
                confidence=1.0,
                source_entity_id=None,
                source_chunk_id=None,
                change_source=ChangeSource.MANUAL,
                ontology_version=graph_sha256(live_path),
                status="approved",
            )
        )

    return new_entries


# ---------------------------------------------------------------------------
# Changelog helpers
# ---------------------------------------------------------------------------

def record_diff(
    changelog: OntologyChangelog,
    version_before: str,
    version_after: str,
    entries_added: list[str],
    entries_removed: list[str],
    source: ChangeSource,
) -> None:
    """Append one entry to the changelog (mutates in place)."""
    changelog.entries.append(
        OntologyChangelogEntry(
            timestamp=_now_iso(),
            ontology_version_before=version_before,
            ontology_version_after=version_after,
            entries_added=entries_added,
            entries_removed=entries_removed,
            change_source=source,
        )
    )


def load_changelog(path: str | Path) -> OntologyChangelog:
    """Load (or create) a changelog from disk."""
    from ontograph.utils.io import load
    p = Path(path)
    if p.exists():
        return load(p, OntologyChangelog)
    # New changelog for a fresh OWL file
    return OntologyChangelog(ontology_path=str(path))


def save_changelog(changelog: OntologyChangelog, path: str | Path) -> None:
    """Persist the changelog to disk."""
    Path(path).write_text(changelog.model_dump_json(indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_entry_id(s: str, p: str, o: str) -> str:
    raw = f"{s}|{p}|{o}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
