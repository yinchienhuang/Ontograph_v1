"""
ontograph.models — shared Pydantic v2 data contracts.

All pipeline stages communicate exclusively through these types.
Import from here rather than from individual submodules so the public
surface stays stable even if files are reorganised.
"""

from ontograph.models.alignment import (
    AlignmentCandidate,
    AlignmentDecision,
    AlignmentMethod,
    OntologyAlignmentBundle,
)
from ontograph.models.document import (
    Chunk,
    DocumentArtifact,
    HeadingNode,
    PageMapEntry,
    RawDocument,
    SourceLocator,
)
from ontograph.models.evaluation import (
    ArmMetrics,
    AttributeChange,
    DesignChangeRequest,
    EvaluationArm,
    EvaluationResult,
    EvidenceSnippet,
)
from ontograph.models.extraction import (
    ExtractedAttribute,
    ExtractedEntity,
    ExtractionBundle,
)
from ontograph.models.ontology import (
    ChangeSource,
    OntologyChangelog,
    OntologyChangelogEntry,
    OntologyDelta,
    OntologyDeltaEntry,
    OntologyTriple,
)
from ontograph.models.synthesis import (
    FactCheckItem,
    ParagraphProvenance,
    SelfCheckResult,
    SynthesizedDocument,
)

__all__ = [
    # document
    "SourceLocator", "HeadingNode", "Chunk", "PageMapEntry",
    "RawDocument", "DocumentArtifact",
    # extraction
    "ExtractedAttribute", "ExtractedEntity", "ExtractionBundle",
    # ontology
    "ChangeSource", "OntologyTriple", "OntologyDeltaEntry", "OntologyDelta",
    "OntologyChangelogEntry", "OntologyChangelog",
    # alignment
    "AlignmentMethod", "AlignmentCandidate", "AlignmentDecision",
    "OntologyAlignmentBundle",
    # synthesis
    "ParagraphProvenance", "FactCheckItem", "SelfCheckResult",
    "SynthesizedDocument",
    # evaluation
    "AttributeChange", "DesignChangeRequest", "EvidenceSnippet",
    "EvaluationArm", "ArmMetrics", "EvaluationResult",
]
