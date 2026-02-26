"""
evaluator/metrics.py — Pydantic models for OWL evaluation results.

An EvaluationReport captures how well a working OWL reconstructed from
pipeline output matches the original source OWL (ground truth).

Two levels of measurement:
  IndividualMetrics — were the right ABox named individuals recovered?
  TripleMetrics     — were the right property assertions recovered?
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Individual-level metrics
# ---------------------------------------------------------------------------

class IndividualMetrics(BaseModel):
    """Coverage of ABox named individuals: source vs working OWL."""

    source_count: int = Field(description="Number of ABox individuals in the source OWL")
    working_count: int = Field(description="Number of ABox individuals in the working OWL")
    recovered_count: int = Field(description="|source ∩ working| — present in both")
    missed_count: int = Field(description="|source - working| — not recovered by pipeline")
    extra_count: int = Field(description="|working - source| — invented by pipeline")

    recall: float = Field(ge=0.0, le=1.0, description="recovered / source_count")
    precision: float = Field(ge=0.0, le=1.0, description="recovered / working_count")
    f1: float = Field(ge=0.0, le=1.0, description="Harmonic mean of recall and precision")


# ---------------------------------------------------------------------------
# Triple-level metrics
# ---------------------------------------------------------------------------

class TripleMetrics(BaseModel):
    """
    Coverage of ABox property assertions.

    Only triples whose subject is an ABox individual are counted — TBox
    class/property declarations are automatically excluded.
    """

    source_count: int = Field(description="ABox triples in source OWL")
    working_count: int = Field(description="ABox triples in working OWL")
    recovered_count: int = Field(description="|source ∩ working| — exact (s,p,o) match")
    missed_count: int = Field(description="|source - working|")
    extra_count: int = Field(description="|working - source|")

    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

class EvaluationReport(BaseModel):
    """
    Full evaluation of a working OWL against a source (ground-truth) OWL.

    Produced by :func:`~ontograph.evaluator.comparator.evaluate` and
    saved to ``data/evaluations/`` for offline analysis.
    """

    id: str = Field(description="sha256(source_owl_path + working_owl_path)[:16]")
    created_at: str = Field(description="ISO 8601 UTC timestamp")
    source_owl_path: str
    working_owl_path: str

    individuals: IndividualMetrics
    triples: TripleMetrics

    missed_individuals: list[str] = Field(
        description="Sorted IRIs present in source but absent from working OWL"
    )
    extra_individuals: list[str] = Field(
        description="Sorted IRIs present in working but absent from source OWL"
    )
