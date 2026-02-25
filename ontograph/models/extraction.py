"""
extraction.py — ExtractionBundle and supporting types.

Represents the output of the extraction step: entities and attributes pulled
from a DocumentArtifact by rule-based and/or LLM-based extractors.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ontograph.models.document import SourceLocator


# ---------------------------------------------------------------------------
# Extracted attribute (a property+value pair on an entity)
# ---------------------------------------------------------------------------

class ExtractedAttribute(BaseModel):
    """
    A single property-value pair extracted for a parent entity.

    Example: name="mass", value="89", unit="kg", raw_text="89 kg"
    """

    name: str = Field(description="Canonical attribute name, e.g. 'mass', 'isp', 'trl_level'")
    raw_text: str = Field(description="Verbatim span from source, e.g. '89 kg'")
    value: str = Field(description="Parsed value, e.g. '89'")
    unit: str | None = Field(default=None, description="Unit string, e.g. 'kg', 's', 'N'")

    # Where this attribute span appears in the source
    chunk_id: str
    source_locator: SourceLocator


# ---------------------------------------------------------------------------
# Extracted relationship (an inter-entity object property)
# ---------------------------------------------------------------------------

class ExtractedRelationship(BaseModel):
    """
    A directed relationship from a parent entity to another named entity.

    Example: OBC1 usesBusProtocol I2C
    Maps to an OWL object property triple (null datatype, IRI object).
    """

    predicate: str = Field(
        description="camelCase property name, e.g. 'usesBusProtocol', 'connectsTo'"
    )
    target: str = Field(
        description="CamelCase local name of the target entity, e.g. 'I2C', 'BatteryPack_A'"
    )
    raw_text: str = Field(
        description="Verbatim span from source, e.g. 'communicates via I2C'"
    )
    chunk_id: str
    source_locator: SourceLocator


# ---------------------------------------------------------------------------
# Extracted entity
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """
    A single named entity extracted from a document chunk.

    Carries its source location (for provenance) and a list of attributes
    (datatype properties) and relationships (object properties) found in
    the same or nearby spans.
    """

    id: str = Field(description="sha256(chunk_id + char_start + text_span[:32])[:16]")
    text_span: str = Field(description="Surface form as it appears in the source")
    chunk_id: str
    source_locator: SourceLocator

    entity_type: str = Field(
        description=(
            "Ontology class name this entity maps to, e.g. 'Subsystem', "
            "'Material', 'Interface', 'Organization', 'Regulation'"
        )
    )
    attributes: list[ExtractedAttribute] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)

    extraction_method: Literal["rule", "llm", "hybrid"] = Field(
        description="Which extractor produced this entity"
    )
    confidence: float = Field(ge=0.0, le=1.0)

    # Section context snapshot from the parent chunk (denormalized for convenience)
    section_context: str = Field(
        default="",
        description="Breadcrumb from parent Chunk.section_context",
    )


# ---------------------------------------------------------------------------
# ExtractionBundle — full output of the extraction step
# ---------------------------------------------------------------------------

class ExtractionBundle(BaseModel):
    """
    All entities extracted from a single DocumentArtifact.

    This is the input to the ontology mapper.
    """

    id: str = Field(description="sha256(document_artifact_id + extractor_version)")
    document_artifact_id: str
    entities: list[ExtractedEntity]

    # Versioning — lets us re-run extraction without ambiguity
    extractor_version: str = Field(
        description=(
            "Semantic version of the rule set + LLM model used, "
            "e.g. '0.1.0/claude-sonnet-4-6'"
        )
    )
    created_at: str = Field(description="ISO 8601 timestamp")

    def entities_by_type(self, entity_type: str) -> list[ExtractedEntity]:
        return [e for e in self.entities if e.entity_type == entity_type]

    def entity_by_id(self, entity_id: str) -> ExtractedEntity | None:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None
