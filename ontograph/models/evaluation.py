"""
evaluation.py — DesignChangeRequest, EvaluationResult, and supporting types.

Represents the input and output of the impact-analysis evaluation module.

Two evaluation arms are always run:
  - docs_only:          retrieves chunks from the synthesized document
  - docs_plus_ontology: same plus 1-hop SPARQL neighbors from the ontology
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Design change request
# ---------------------------------------------------------------------------

class AttributeChange(BaseModel):
    """A single property value that changes in the design modification."""

    property_iri: str = Field(description="OWL datatype property IRI, e.g. 'aero:hasMass'")
    old_value: str
    new_value: str
    unit: str | None = None


class DesignChangeRequest(BaseModel):
    """
    Describes a proposed modification to a design component.

    `component_iri` is optional: if the ontology is available the caller
    provides the IRI so the docs+ontology arm can run SPARQL on it.
    """

    id: str = Field(description="sha256(description + created_at)[:16]")
    description: str = Field(
        description="Free-text narrative of the change, used for embedding retrieval"
    )
    component_iri: str | None = Field(
        default=None,
        description="OWL instance IRI of the changed component (if ontology available)",
    )
    attribute_changes: list[AttributeChange] = Field(default_factory=list)
    created_at: str = Field(description="ISO 8601 timestamp")


# ---------------------------------------------------------------------------
# Evaluation arm
# ---------------------------------------------------------------------------

class EvidenceSnippet(BaseModel):
    """One piece of evidence cited by the LLM reasoner."""

    source: Literal["document_chunk", "ontology_extracted", "ontology_organizational"]
    ref_id: str = Field(description="Chunk.id or OntologyDeltaEntry.id")
    text: str = Field(description="The snippet text as presented to the LLM")
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="Cosine similarity to the change description embedding",
    )


class EvaluationArm(BaseModel):
    """
    The result of one retrieval+reasoning pass.

    `predicted_impacted_items` is the LLM's answer — a list of component
    names / subsystem IRIs it believes are affected by the design change.
    """

    name: Literal["docs_only", "docs_plus_ontology"]
    predicted_impacted_items: list[str]
    reasoning_trace: str = Field(
        description="Full LLM reasoning output (structured JSON narrative)"
    )
    evidence: list[EvidenceSnippet]


# ---------------------------------------------------------------------------
# Per-arm metrics
# ---------------------------------------------------------------------------

class ArmMetrics(BaseModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(
        ge=0.0, le=1.0,
        description="Mean relevance_score across all cited EvidenceSnippets",
    )


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """
    Full output of one evaluation run comparing docs_only vs docs+ontology arms.

    `ground_truth_impacted_items` is provided by the user (subject-matter expert)
    and is the reference set for precision/recall/F1 calculation.
    """

    id: str = Field(description="sha256(design_change_request_id + synthesized_document_id)[:16]")
    design_change_request_id: str
    synthesized_document_id: str

    ground_truth_impacted_items: list[str] = Field(
        description="Expert-labeled list of items actually impacted by this change"
    )

    arms: list[EvaluationArm]
    metrics: dict[str, ArmMetrics] = Field(
        description="arm name → ArmMetrics, e.g. {'docs_only': ..., 'docs_plus_ontology': ...}"
    )

    created_at: str = Field(description="ISO 8601 timestamp")

    def arm(self, name: Literal["docs_only", "docs_plus_ontology"]) -> EvaluationArm | None:
        for a in self.arms:
            if a.name == name:
                return a
        return None

    def winner(self) -> str | None:
        """Returns the arm name with the higher F1, or None if tied / missing."""
        if len(self.metrics) < 2:
            return None
        ranked = sorted(self.metrics.items(), key=lambda kv: kv[1].f1, reverse=True)
        if ranked[0][1].f1 == ranked[1][1].f1:
            return None
        return ranked[0][0]
