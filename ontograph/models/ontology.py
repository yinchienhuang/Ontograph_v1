"""
ontology.py — OntologyDelta, OntologyChangelog, and supporting types.

Represents proposed and approved changes to the working OWL ABox, along with
a full audit trail that lets us reconcile manual edits made outside the
pipeline (e.g., in Protégé).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Change source — who/what created a triple
# ---------------------------------------------------------------------------

class ChangeSource(str, Enum):
    PIPELINE  = "pipeline"   # written by mapper.py
    ALIGNMENT = "alignment"  # rewritten by aligner.py (canonical IRI + aliases)
    MANUAL    = "manual"     # detected by owl_diff.py after a human edit
    REVIEWER  = "reviewer"   # human changed the value during CLI review


# ---------------------------------------------------------------------------
# OWL triple representation
# ---------------------------------------------------------------------------

class OntologyTriple(BaseModel):
    """
    A single RDF triple using full IRIs or XSD-typed literals.

    subject and predicate must always be IRIs.
    object is an IRI for object properties, a plain string for datatype/annotation
    properties (use datatype to indicate the XSD type).
    """

    subject: str = Field(description="Full IRI, e.g. 'aero:ThrusterModule_42'")
    predicate: str = Field(description="Full IRI, e.g. 'aero:hasMass'")
    object: str = Field(
        description=(
            "Full IRI for object properties; "
            "plain string literal for datatype properties"
        )
    )
    datatype: str | None = Field(
        default=None,
        description="XSD datatype IRI for literals, e.g. 'xsd:float'",
    )
    language: str | None = Field(
        default=None,
        description="BCP-47 language tag for plain string literals, e.g. 'en'",
    )


# ---------------------------------------------------------------------------
# OntologyDeltaEntry — one proposed/approved triple change
# ---------------------------------------------------------------------------

class OntologyDeltaEntry(BaseModel):
    """
    A single triple to be added to the working ontology, with full provenance.

    The `status` field drives the CLI review loop:
      proposed  → shown to reviewer
      approved  → written to OWL
      rejected  → skipped, recorded for audit
      edited    → reviewer changed object/datatype before approval
    """

    id: str = Field(description="sha256(triple fields concatenated)[:16]")
    triple: OntologyTriple

    rationale: str = Field(
        description="Why this triple was inferred (from LLM or rule engine)"
    )
    confidence: float = Field(ge=0.0, le=1.0)

    # Provenance — may be None for MANUAL entries
    source_entity_id: str | None = None
    source_chunk_id: str | None = None

    change_source: ChangeSource = ChangeSource.PIPELINE

    # Version hash of the OWL file at the time this entry was written.
    # Lets owl_diff.py detect whether subsequent manual edits conflict.
    ontology_version: str | None = Field(
        default=None,
        description="sha256 of working OWL file bytes at write time",
    )

    status: Literal["proposed", "approved", "rejected", "edited"] = "proposed"
    reviewer_note: str | None = None
    reviewed_at: str | None = None


# ---------------------------------------------------------------------------
# OntologyDelta — all entries for one extraction run
# ---------------------------------------------------------------------------

class OntologyDelta(BaseModel):
    """
    The full set of proposed ontology changes produced from one ExtractionBundle.

    Passes through three stages (data/deltas/proposed → aligned → approved)
    before triples are written to OWL.
    """

    id: str = Field(description="sha256(extraction_bundle_id + base_ontology_iri)")
    extraction_bundle_id: str
    base_ontology_iri: str = Field(
        description="IRI of the TBox (class/property definitions) being extended"
    )
    entries: list[OntologyDeltaEntry]
    created_at: str = Field(description="ISO 8601 timestamp")

    def approved_entries(self) -> list[OntologyDeltaEntry]:
        return [e for e in self.entries if e.status == "approved"]

    def pending_entries(self) -> list[OntologyDeltaEntry]:
        return [e for e in self.entries if e.status == "proposed"]

    def entry_by_id(self, entry_id: str) -> OntologyDeltaEntry | None:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None


# ---------------------------------------------------------------------------
# Ontology changelog — append-only audit trail per working OWL file
# ---------------------------------------------------------------------------

class OntologyChangelogEntry(BaseModel):
    """Records one write transaction to the working OWL file."""

    timestamp: str
    ontology_version_before: str = Field(description="sha256 of OWL before write")
    ontology_version_after: str = Field(description="sha256 of OWL after write")
    entries_added: list[str] = Field(
        default_factory=list,
        description="OntologyDeltaEntry.id list written in this transaction",
    )
    entries_removed: list[str] = Field(
        default_factory=list,
        description="Triple IRIs retracted in this transaction (rare)",
    )
    change_source: ChangeSource


class OntologyChangelog(BaseModel):
    """
    Append-only audit trail for one working OWL file.

    Stored at data/ontology/changelog/<filename>.json.
    owl_diff.py uses this to identify triples added outside the pipeline.
    """

    ontology_path: str = Field(description="Absolute path to the working OWL file")
    entries: list[OntologyChangelogEntry] = Field(default_factory=list)

    def latest_version(self) -> str | None:
        if not self.entries:
            return None
        return self.entries[-1].ontology_version_after
