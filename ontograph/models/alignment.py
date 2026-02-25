"""
alignment.py — OntologyAlignmentBundle and supporting types.

Represents the entity-resolution step: detecting that two entity mentions
(e.g., "FAA" and "Federal Aviation Administration") refer to the same
real-world concept and should share a single OWL IRI.

Pipeline position:
    OntologyDelta (proposed) → aligner.py → OntologyAlignmentBundle
    → merged OntologyDelta (aligned) → reviewer
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AlignmentMethod(str, Enum):
    ACRONYM           = "acronym"       # rule: known acronym table
    STRING_SIMILARITY = "string_sim"    # token-overlap / edit distance
    LLM               = "llm"           # LLM judgment call
    MANUAL            = "manual"        # human forced merge via CLI


# ---------------------------------------------------------------------------
# AlignmentCandidate — a (possibly same) entity pair, before decision
# ---------------------------------------------------------------------------

class AlignmentCandidate(BaseModel):
    """
    A pair of extracted entities flagged as potentially the same concept.

    The aligner generates candidates; a decision is made for each one.
    """

    id: str = Field(description="sha256(entity_id_a + entity_id_b)[:16]")
    entity_id_a: str = Field(description="ExtractedEntity.id")
    entity_id_b: str = Field(description="ExtractedEntity.id")
    surface_a: str = Field(description="Surface form of entity A, e.g. 'FAA'")
    surface_b: str = Field(
        description="Surface form of entity B, e.g. 'Federal Aviation Administration'"
    )
    similarity_score: float = Field(ge=0.0, le=1.0)
    method: AlignmentMethod
    rationale: str = Field(description="Why this pair was flagged")


# ---------------------------------------------------------------------------
# AlignmentDecision — resolved outcome for one candidate pair
# ---------------------------------------------------------------------------

class AlignmentDecision(BaseModel):
    """
    The resolution of one AlignmentCandidate.

    On approval:
    - `canonical_entity_id` is kept; its OWL IRI is used going forward.
    - All `merged_entity_ids` have their OntologyDeltaEntry IRIs rewritten to
      the canonical IRI.
    - All surface forms in `aliases` are written as `skos:altLabel` triples.
    """

    candidate_id: str
    canonical_entity_id: str = Field(
        description="The surviving ExtractedEntity.id (usually the formal name)"
    )
    merged_entity_ids: list[str] = Field(
        description="Entity IDs absorbed into the canonical node"
    )
    aliases: list[str] = Field(
        description="All surface forms to record as skos:altLabel"
    )
    status: Literal["proposed", "approved", "rejected"] = "proposed"
    reviewer_note: str | None = None


# ---------------------------------------------------------------------------
# OntologyAlignmentBundle — full alignment run result
# ---------------------------------------------------------------------------

class OntologyAlignmentBundle(BaseModel):
    """
    All candidate pairs and their decisions for one OntologyDelta run.

    After decisions are approved, the aligner rewrites the delta in-place
    (updating IRIs and inserting altLabel triples) before it goes to the
    human reviewer.
    """

    id: str = Field(description="sha256(ontology_delta_id + 'alignment')[:16]")
    ontology_delta_id: str
    candidates: list[AlignmentCandidate] = Field(default_factory=list)
    decisions: list[AlignmentDecision] = Field(default_factory=list)
    created_at: str = Field(description="ISO 8601 timestamp")

    def approved_decisions(self) -> list[AlignmentDecision]:
        return [d for d in self.decisions if d.status == "approved"]

    def pending_decisions(self) -> list[AlignmentDecision]:
        return [d for d in self.decisions if d.status == "proposed"]

    def decision_for(self, candidate_id: str) -> AlignmentDecision | None:
        for d in self.decisions:
            if d.candidate_id == candidate_id:
                return d
        return None
