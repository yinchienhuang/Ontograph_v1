"""
ontograph/reconstruction/schema.py — Data models for triple reconstruction evaluation.

Two approaches are compared:
  - ontograph arm: full pipeline (chunk → extract → map → align → working OWL)
  - direct arm:    single LLM call on raw document text (no chunking)

Results are scored by comparing reconstructed triples against the source OWL (ground truth).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DirectTriple(BaseModel):
    """One extracted fact from the direct LLM arm."""

    subject: str            # IRI-safe local name the LLM assigned to the individual
    rdf_type: str | None    # class local name from TBox (e.g. "OnboardComputer"); None if unknown
    property: str           # property local name, e.g. "powerW"
    value: str              # plain string value (no units), e.g. "2.0"


class DirectExtractionResult(BaseModel):
    """Structured output of the direct LLM extraction call."""

    triples: list[DirectTriple]


class TripleDetail(BaseModel):
    """One (subject, predicate, object) triple as plain strings for debug output."""
    subject: str
    predicate: str
    object: str


class ArmDebug(BaseModel):
    """Per-arm breakdown of matched / missed / extra individuals and triples."""

    arm: Literal["ontograph", "direct"]

    matched_individuals: list[str]   # IRIs recovered by this arm (source ∩ predicted)
    missed_individuals: list[str]    # IRIs in source but absent from prediction
    extra_individuals: list[str]     # IRIs in prediction but absent from source

    matched_triples: list[TripleDetail]  # exact (s,p,o) matches
    missed_triples: list[TripleDetail]   # in source, not predicted
    extra_triples: list[TripleDetail]    # in prediction, not in source


class ReconstructionDebug(BaseModel):
    """
    Detailed per-triple debug companion to ReconstructionReport.

    Saved alongside the summary report as ``<id>_debug.json``.
    """

    model_config = {"protected_namespaces": ()}

    id: str            # same id as the parent ReconstructionReport
    created_at: str
    source_owl_path: str
    arms: list[ArmDebug]


class ArmResult(BaseModel):
    """Evaluation metrics for one reconstruction arm (ontograph or direct)."""

    arm: Literal["ontograph", "direct"]

    individual_precision: float
    individual_recall: float
    individual_f1: float

    triple_precision: float
    triple_recall: float
    triple_f1: float

    triple_count_source: int       # ABox triples in the ground-truth OWL
    triple_count_predicted: int    # ABox triples produced by this arm


class ReconstructionReport(BaseModel):
    """
    Full comparison report for one reconstruction experiment.

    Saved to data/evaluations/ by run_reconstruction().
    """

    model_config = {"protected_namespaces": ()}

    id: str
    created_at: str
    source_owl_path: str
    document_path: str
    provider: str
    arms: list[ArmResult]       # [ontograph_arm, direct_arm] when mode=="both"
    winner: str | None          # arm name with higher triple_f1; None if tied or single arm
