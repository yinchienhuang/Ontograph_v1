"""
ontograph/reconstruction — Triple reconstruction comparison: onto-graph vs. direct AI.

Compares two approaches for reconstructing OWL triples from a document:
  - ontograph arm: full chunked pipeline (extract → map → align → OWL)
  - direct arm:    single LLM call on the raw document text

Both arms are scored against the source OWL (ground truth) and the results
are captured in a ReconstructionReport.

Usage:
    from ontograph.reconstruction import run_reconstruction
    from ontograph.llm import get_provider

    provider = get_provider("claude")
    report = run_reconstruction(
        source_owl="data/ontology/cubesatontology.owl",
        document_path="data/raw/cubesatontology_synthesized.md",
        provider=provider,
        mode="both",
    )
    print(f"Winner: {report.winner}")
"""

from ontograph.reconstruction.schema import (
    DirectTriple,
    DirectExtractionResult,
    TripleDetail,
    ArmDebug,
    ReconstructionDebug,
    ArmResult,
    ReconstructionReport,
)
from ontograph.reconstruction.direct_extractor import extract_direct
from ontograph.reconstruction.runner import run_reconstruction, pick_winner

__all__ = [
    "DirectTriple",
    "DirectExtractionResult",
    "TripleDetail",
    "ArmDebug",
    "ReconstructionDebug",
    "ArmResult",
    "ReconstructionReport",
    "extract_direct",
    "run_reconstruction",
    "pick_winner",
]
