"""
synthesis.py — SynthesizedDocument and supporting types.

Represents the output of the document synthesizer: a grounded Markdown report
generated from the approved OWL triples, with full paragraph-level provenance
and a self-check result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paragraph-level provenance
# ---------------------------------------------------------------------------

class ParagraphProvenance(BaseModel):
    """
    Maps one paragraph in the generated Markdown to the OWL triples it encodes.

    `citation_anchors` are the inline markers embedded in the Markdown text,
    e.g. "[T-042]", so readers can correlate text to triples visually.
    """

    paragraph_index: int = Field(description="0-indexed position in the document")
    triple_ids: list[str] = Field(
        description="OntologyDeltaEntry.id references encoded by this paragraph"
    )
    citation_anchors: list[str] = Field(
        description="Inline markers embedded in the Markdown, e.g. ['[T-042]', '[T-043]']"
    )


# ---------------------------------------------------------------------------
# Self-check result
# ---------------------------------------------------------------------------

class FactCheckItem(BaseModel):
    """Comparison of one expected triple against what was found in the generated text."""

    triple_id: str = Field(description="OntologyDeltaEntry.id")
    expected_object: str = Field(description="Literal or IRI from the ontology")
    found_text: str | None = Field(
        default=None,
        description="Re-extracted value from the generated text (None if not found)",
    )
    match: bool = Field(
        description=(
            "True if found_text matches expected_object "
            "(exact for literals ≤ 1% numeric tolerance)"
        )
    )
    note: str | None = None


class SelfCheckResult(BaseModel):
    """
    Summary of the self-check pass: re-extract facts from generated text and
    compare to the source ontology triples.

    A coverage < 1.0 means some triples were not faithfully reflected in the
    generated document and should be reviewed before the document is published.
    """

    checked_triple_count: int
    matched_count: int
    coverage: float = Field(
        ge=0.0, le=1.0,
        description="matched_count / checked_triple_count",
    )
    items: list[FactCheckItem]

    @property
    def discrepancies(self) -> list[FactCheckItem]:
        return [item for item in self.items if not item.match]


# ---------------------------------------------------------------------------
# SynthesizedDocument
# ---------------------------------------------------------------------------

class SynthesizedDocument(BaseModel):
    """
    A grounded natural-language document generated from an approved OntologyDelta.

    Every paragraph is tied to the triples it encodes via `provenance`.
    No facts may appear in the Markdown that are not in the source ontology
    (enforced by the self-check step).
    """

    id: str = Field(description="sha256(ontology_delta_id + title)[:16]")
    ontology_delta_id: str

    title: str
    markdown: str = Field(description="Full generated document in GitHub-flavored Markdown")

    provenance: list[ParagraphProvenance] = Field(
        default_factory=list,
        description="One entry per paragraph; covers all triples cited",
    )

    # Populated after the self-check step; None if not yet run
    self_check: SelfCheckResult | None = None

    created_at: str = Field(description="ISO 8601 timestamp")

    def triples_cited(self) -> set[str]:
        """Return the set of all triple IDs cited anywhere in the document."""
        return {tid for p in self.provenance for tid in p.triple_ids}

    def is_fully_grounded(self) -> bool:
        """True if self-check passed with 100% coverage."""
        return self.self_check is not None and self.self_check.coverage == 1.0
