"""
ingest/extractor.py — Extract entities and attributes from document chunks.

For each Chunk in a DocumentArtifact one LLM call is made.  The model is
instructed to return a structured list of named entities (components,
subsystems, materials, …) together with their measured attributes.

Pipeline position:
    DocumentArtifact  →  extract()  →  ExtractionBundle
                                           ↓
                                       mapper.py
"""

from __future__ import annotations

import hashlib
import warnings
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ontograph.llm.base import LLMMessage, LLMProvider, LLMRequest
from ontograph.models.document import Chunk, DocumentArtifact
from ontograph.models.extraction import (
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionBundle,
)
from ontograph.utils.owl import TBoxSummary

# ---------------------------------------------------------------------------
# LLM response schemas
# ---------------------------------------------------------------------------

class RawAttribute(BaseModel):
    """One property-value pair as the LLM extracted it."""

    name: str = Field(
        description=(
            "Snake_case attribute name, e.g. 'dry_mass', 'specific_impulse', 'trl_level'"
        )
    )
    raw_text: str = Field(description="Verbatim span from source, e.g. '12.4 kg'")
    value: str = Field(description="Parsed numeric or string value, e.g. '12.4'")
    unit: str | None = Field(
        default=None,
        description="Unit string if present, e.g. 'kg', 's', 'N', 'bar'",
    )


class RawRelationship(BaseModel):
    """One inter-entity relationship as the LLM extracted it."""

    predicate: str = Field(
        description=(
            "camelCase property name, e.g. 'usesBusProtocol', 'connectsTo', "
            "'hasPropellant', 'interfacesWith'"
        )
    )
    target: str = Field(
        description=(
            "CamelCase local name of the target entity, no spaces "
            "(e.g. 'I2C', 'SPI', 'BatteryPack_A', 'EPS')"
        )
    )
    raw_text: str = Field(
        description="Verbatim span from source, e.g. 'communicates via I2C'"
    )


class RawEntity(BaseModel):
    """One entity extracted from a chunk."""

    text_span: str = Field(
        description="Exact surface form as it appears in the source text"
    )
    entity_type: str = Field(
        description=(
            "One of: Component, Subsystem, Material, Interface, "
            "Organization, Standard"
        )
    )
    confidence: float = Field(ge=0.0, le=1.0)
    attributes: list[RawAttribute] = Field(default_factory=list)
    relationships: list[RawRelationship] = Field(default_factory=list)


class ChunkExtractionResponse(BaseModel):
    """Structured output returned by the LLM for one chunk."""

    entities: list[RawEntity] = Field(
        description="All named entities found in this document section"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SYSTEM_HEADER = """\
You are an aerospace knowledge extraction engine.

Extract named entities from the aerospace document section provided.

ENTITY TYPES (use exactly one per entity):
"""

_SYSTEM_FALLBACK_TYPES = """\
  Component    — a physical hardware item (thruster, valve, sensor, tank …)
  Subsystem    — a functional grouping (propulsion subsystem, ADCS, power …)
  Material     — a substance or material (titanium alloy, MON-3/MMH …)
  Interface    — a connection between two systems
  Organization — a company, agency, or authority (NASA, FAA, ESA …)
  Standard     — a standard or specification (MIL-STD-1553, ECSS-E-ST-10C …)\
"""

_SYSTEM_FOOTER = """

FOR EACH ENTITY extract every measurable attribute mentioned in the text:
  name      — snake_case canonical name (e.g. "dry_mass", "specific_impulse")
  raw_text  — verbatim span (e.g. "12.4 kg")
  value     — parsed value without unit (e.g. "12.4")
  unit      — unit string or null (e.g. "kg")

FOR EACH ENTITY also extract relationships to other named entities:
  predicate — camelCase property (e.g. "usesBusProtocol", "connectsTo", "hasPropellant")
  target    — CamelCase local name of the target entity, no spaces (e.g. "I2C", "SPI", "EPS")
  raw_text  — verbatim span (e.g. "communicates via I2C")

ATTRIBUTES vs RELATIONSHIPS:
  - Attribute: the value is a number, unit, or descriptive string ("12.4 kg", "titanium")
  - Relationship: the value is another named entity in the document ("I2C bus", "battery pack")
  - Protocols, buses, referenced components/subsystems, and propellants are relationships.

RULES:
  - Only extract entities, attributes, and relationships explicitly stated in the text.
  - Do NOT infer or hallucinate values not present in the source.
  - If no entities are found, return an empty list.
  - Return ONLY valid JSON matching the required schema. No preamble.
"""


def _build_system_prompt(tbox: TBoxSummary | None) -> str:
    """Build the extraction system prompt, optionally injecting TBox class names."""
    if tbox and tbox.classes:
        classes_block = "\n".join(f"  {c}" for c in sorted(tbox.classes))
        type_section = (
            classes_block
            + "\n\n  If no class fits, use the closest generic type: "
            "Component, Subsystem, Location, Person, or Operation."
        )
    else:
        type_section = _SYSTEM_FALLBACK_TYPES
    return _SYSTEM_HEADER + type_section + _SYSTEM_FOOTER


def _build_extraction_messages(chunk: Chunk, tbox: TBoxSummary | None = None) -> list[LLMMessage]:
    user_content = (
        f"Document section:\n\n"
        f"{chunk.to_llm_context()}\n\n"
        f"Extract all named entities and their attributes from this text."
    )
    return [
        LLMMessage(role="system", content=_build_system_prompt(tbox)),
        LLMMessage(role="user",   content=user_content),
    ]


def _make_entity_id(chunk_id: str, text_span: str) -> str:
    raw = f"{chunk_id}:{text_span[:32]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _convert_entity(raw: RawEntity, chunk: Chunk) -> ExtractedEntity:
    attributes = [
        ExtractedAttribute(
            name=attr.name,
            raw_text=attr.raw_text,
            value=attr.value,
            unit=attr.unit,
            chunk_id=chunk.id,
            source_locator=chunk.source_locator,
        )
        for attr in raw.attributes
    ]
    relationships = [
        ExtractedRelationship(
            predicate=rel.predicate,
            target=rel.target,
            raw_text=rel.raw_text,
            chunk_id=chunk.id,
            source_locator=chunk.source_locator,
        )
        for rel in raw.relationships
    ]
    return ExtractedEntity(
        id=_make_entity_id(chunk.id, raw.text_span),
        text_span=raw.text_span,
        chunk_id=chunk.id,
        source_locator=chunk.source_locator,
        entity_type=raw.entity_type,
        attributes=attributes,
        relationships=relationships,
        extraction_method="llm",
        confidence=raw.confidence,
        section_context=chunk.section_context,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(
    artifact: DocumentArtifact,
    provider: LLMProvider,
    min_chunk_chars: int = 30,
    tbox: TBoxSummary | None = None,
) -> ExtractionBundle:
    """
    Extract entities and attributes from every qualifying chunk.

    One LLM call is made per chunk.  Chunks shorter than *min_chunk_chars*
    (typically headings with no body text) are silently skipped.

    If a single chunk's LLM call fails a warning is emitted and the chunk is
    skipped so the rest of the document is still processed.

    Args:
        artifact:        Output of :func:`~ontograph.ingest.chunker.chunk`.
        provider:        Any :class:`~ontograph.llm.base.LLMProvider`.
        min_chunk_chars: Skip chunks with fewer characters than this.

    Returns:
        An :class:`~ontograph.models.extraction.ExtractionBundle`.
    """
    all_entities: list[ExtractedEntity] = []

    for c in artifact.chunks:
        if len(c.text) < min_chunk_chars:
            continue

        messages = _build_extraction_messages(c, tbox=tbox)
        request = LLMRequest(
            messages=messages,
            response_model=ChunkExtractionResponse,
            temperature=0.0,   # deterministic for extraction
            max_tokens=4096,
        )

        try:
            response = provider.complete(request)
        except Exception as exc:
            warnings.warn(
                f"Extraction failed for chunk {c.id} "
                f"(section: {c.section_context}): {exc}",
                stacklevel=2,
            )
            continue

        extraction: ChunkExtractionResponse = response.parsed
        for raw_entity in extraction.entities:
            all_entities.append(_convert_entity(raw_entity, c))

    version = f"0.1.0/{provider.model_id}"
    bundle_id = hashlib.sha256(
        f"{artifact.id}:{version}".encode()
    ).hexdigest()[:16]

    return ExtractionBundle(
        id=bundle_id,
        document_artifact_id=artifact.id,
        entities=all_entities,
        extractor_version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
