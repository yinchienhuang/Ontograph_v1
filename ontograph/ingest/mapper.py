"""
ingest/mapper.py — Map an ExtractionBundle to a proposed OntologyDelta.

For each extracted entity one LLM call is made.  The model proposes:
  - a normalised subject local name (e.g. "ThrusterModule_A")
  - the OWL class (rdf:type)
  - a predicate + object + datatype for every attribute

Subject and predicate local names are expanded to full IRIs using the
supplied *namespace*.  The ``xsd:`` prefix in datatypes is expanded to
``http://www.w3.org/2001/XMLSchema#``.

Pipeline position:
    ExtractionBundle  →  map_to_delta()  →  OntologyDelta (status="proposed")
                                                 ↓
                                         reviewer CLI
"""

from __future__ import annotations

import hashlib
import re
import warnings
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ontograph.llm.base import LLMMessage, LLMProvider, LLMRequest
from ontograph.models.extraction import ExtractedEntity, ExtractionBundle
from ontograph.models.ontology import (
    ChangeSource,
    OntologyDelta,
    OntologyDeltaEntry,
    OntologyTriple,
)
from ontograph.utils.owl import TBoxSummary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RDF_TYPE   = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_CMT   = "http://www.w3.org/2000/01/rdf-schema#comment"
_XSD_BASE   = "http://www.w3.org/2001/XMLSchema#"

DEFAULT_NAMESPACE = "http://example.org/aerospace#"

_CAMEL_SEP = re.compile(r"[_\s]+")


# ---------------------------------------------------------------------------
# LLM response schemas
# ---------------------------------------------------------------------------

class ProposedTriple(BaseModel):
    """One OWL triple proposed by the mapper LLM."""

    predicate: str = Field(
        description=(
            "camelCase property local name, e.g. 'hasDryMass', 'hasSpecificImpulse'. "
            "Use 'rdf:type' only for the class assertion."
        )
    )
    object: str = Field(
        description=(
            "For datatype properties: the literal value without unit (e.g. '12.4'). "
            "For object properties: the target entity's CamelCase local name "
            "(e.g. 'I2C', 'BatteryPack_A', 'PropulsionSubsystem')."
        )
    )
    datatype: str | None = Field(
        default=None,
        description=(
            "'xsd:float' for decimals, 'xsd:integer' for whole numbers, "
            "null for IRI objects or plain strings"
        ),
    )
    rationale: str = Field(description="One-sentence justification")
    confidence: float = Field(ge=0.0, le=1.0)


class EntityMappingResponse(BaseModel):
    """Structured output returned by the LLM for one entity."""

    subject_local_name: str = Field(
        description=(
            "Normalised local name for the OWL instance, e.g. 'ThrusterModule_A'. "
            "CamelCase, underscores allowed, no spaces."
        )
    )
    rdf_type: str = Field(
        description=(
            "OWL class local name, e.g. 'PropulsionSubsystem', 'PropellantTank'. "
            "Prefer specific aerospace classes over generic ones."
        )
    )
    description: str = Field(
        default="",
        description=(
            "One or two sentences describing what this specific entity is and "
            "its role in the system, based on the extracted text. "
            "Will be stored as rdfs:comment on the OWL individual."
        ),
    )
    triples: list[ProposedTriple] = Field(
        description=(
            "One triple per attribute (datatype property) plus one triple per "
            "relationship (object property). Must cover every supplied attribute "
            "and every supplied relationship."
        )
    )


# ---------------------------------------------------------------------------
# IRI helpers
# ---------------------------------------------------------------------------

def _expand_datatype(dt: str | None) -> str | None:
    """Expand 'xsd:Foo' → full XSD IRI.  Full IRIs are returned unchanged."""
    if not dt:
        return None
    if dt.startswith("http://") or dt.startswith("https://"):
        return dt
    if dt.startswith("xsd:"):
        return _XSD_BASE + dt[4:]
    return None


def _iri(local: str, namespace: str) -> str:
    """
    Resolve a local name to a full IRI.

    Resolution order (first match wins):
      1. Already a full IRI (starts with http:// / https://) → pass through
      2. Special prefixes: rdf:type, xsd:* → expand
      3. Anything else → prepend namespace after stripping non-word chars
    """
    if local.startswith("http://") or local.startswith("https://"):
        return local
    if local == "rdf:type":
        return _RDF_TYPE
    if local.startswith("xsd:"):
        return _XSD_BASE + local[4:]
    clean = re.sub(r"[^\w]+", "_", local).strip("_")
    return f"{namespace}{clean}"


def _entry_id(subject: str, predicate: str, obj: str) -> str:
    """Stable ID for a triple: sha256(s|p|o)[:16]."""
    raw = f"{subject}|{predicate}|{obj}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are an OWL ontology mapper for aerospace systems engineering.

Convert an extracted entity, its attributes, and its relationships into OWL ABox triples.

NAMING RULES:
  subject_local_name — CamelCase with underscores for identifiers
    • Keep existing identifiers intact (ThrusterModule_A → ThrusterModule_A)
    • Normalise spaces to CamelCase (thruster module A → ThrusterModule_A)
  rdf_type — most specific OWL class possible
    • Avoid generic: Thing, Entity, Object
  description — 1-2 sentences describing what this specific entity is and its \
role in the system, drawn from the source text. This becomes rdfs:comment.
  predicate — camelCase, e.g. hasDryMass, usesBusProtocol, connectsTo
  datatype — "xsd:float" for decimals, "xsd:integer" for integers, null for object properties
  object — literal value (no unit) for datatype properties, or CamelCase local name for object properties

DATATYPE vs OBJECT PROPERTIES:
  - Attribute (datatype property): value is a number or string literal → set datatype
  - Relationship (object property): value is another named entity → null datatype, target as local name

COMPLETENESS:
  Every attribute AND every relationship in the input MUST produce exactly one triple.

Return ONLY valid JSON matching the schema. No preamble or explanation.
"""


def _build_system_prompt(tbox: TBoxSummary | None) -> str:
    """Compose the system prompt, optionally appending TBox vocabulary."""
    if tbox is None:
        return _SYSTEM_PROMPT_BASE
    tbox_block = (
        "\n\nTBOX VOCABULARY — prefer these names over invented ones:\n"
        + tbox.to_prompt_block()
        + "\n\nIf the entity fits a listed class, use it for rdf_type. "
        "If an attribute matches a listed datatype property, use that predicate name. "
        "Only invent new names when there is no suitable match in the vocabulary above."
    )
    return _SYSTEM_PROMPT_BASE + tbox_block


def _build_mapping_messages(
    entity: ExtractedEntity,
    namespace: str,
    tbox: TBoxSummary | None = None,
) -> list[LLMMessage]:
    attr_lines = "\n".join(
        f"  - {a.name}: raw='{a.raw_text}', value='{a.value}'"
        + (f", unit='{a.unit}'" if a.unit else "")
        for a in entity.attributes
    ) or "  (none)"

    rel_lines = "\n".join(
        f"  - {r.predicate}: target='{r.target}' (raw: '{r.raw_text}')"
        for r in entity.relationships
    ) or "  (none)"

    user_content = (
        f"Namespace: {namespace}\n"
        f"Section context: {entity.section_context or '(root)'}\n\n"
        f"Entity surface form : {entity.text_span}\n"
        f"Entity type (hint)  : {entity.entity_type}\n\n"
        f"Extracted attributes (datatype properties):\n{attr_lines}\n\n"
        f"Relationships to other entities (object properties — null datatype, "
        f"target as CamelCase local name):\n{rel_lines}\n\n"
        f"Map this entity to OWL triples. "
        f"Every attribute must produce one datatype triple. "
        f"Every relationship must produce one object property triple (null datatype)."
    )

    return [
        LLMMessage(role="system", content=_build_system_prompt(tbox)),
        LLMMessage(role="user",   content=user_content),
    ]


# ---------------------------------------------------------------------------
# Conversion: EntityMappingResponse → OntologyDeltaEntry list
# ---------------------------------------------------------------------------

def _convert_mapping(
    mapping: EntityMappingResponse,
    entity: ExtractedEntity,
    namespace: str,
) -> list[OntologyDeltaEntry]:
    subject_iri = _iri(mapping.subject_local_name, namespace)
    entries: list[OntologyDeltaEntry] = []

    # rdf:type triple (always first)
    type_iri = _iri(mapping.rdf_type, namespace)
    type_triple = OntologyTriple(
        subject=subject_iri,
        predicate=_RDF_TYPE,
        object=type_iri,
    )
    entries.append(OntologyDeltaEntry(
        id=_entry_id(subject_iri, _RDF_TYPE, type_iri),
        triple=type_triple,
        rationale=f"Entity type inferred from extracted entity '{entity.text_span}'",
        confidence=entity.confidence,
        source_entity_id=entity.id,
        source_chunk_id=entity.chunk_id,
        change_source=ChangeSource.PIPELINE,
        status="proposed",
    ))

    # rdfs:comment — natural-language description of the individual
    if mapping.description.strip():
        cmt_triple = OntologyTriple(
            subject=subject_iri,
            predicate=_RDFS_CMT,
            object=mapping.description.strip(),
            language="en",
        )
        entries.append(OntologyDeltaEntry(
            id=_entry_id(subject_iri, _RDFS_CMT, mapping.description.strip()),
            triple=cmt_triple,
            rationale="Description of this individual derived from source text",
            confidence=entity.confidence,
            source_entity_id=entity.id,
            source_chunk_id=entity.chunk_id,
            change_source=ChangeSource.PIPELINE,
            status="proposed",
        ))

    # Attribute triples
    for pt in mapping.triples:
        if pt.predicate in ("rdf:type", "type"):
            continue  # already handled above

        predicate_iri = _iri(pt.predicate, namespace)
        dt = _expand_datatype(pt.datatype)

        # Decide whether the object is a literal or an IRI
        if dt:
            obj_str = pt.object
        else:
            # If it looks like a bare local name (no spaces, no special chars),
            # treat as IRI reference; otherwise it's a plain string literal.
            if re.match(r"^[\w_]+$", pt.object):
                obj_str = _iri(pt.object, namespace)
            else:
                obj_str = pt.object

        triple = OntologyTriple(
            subject=subject_iri,
            predicate=predicate_iri,
            object=obj_str,
            datatype=dt,
        )
        entries.append(OntologyDeltaEntry(
            id=_entry_id(subject_iri, predicate_iri, obj_str),
            triple=triple,
            rationale=pt.rationale,
            confidence=pt.confidence,
            source_entity_id=entity.id,
            source_chunk_id=entity.chunk_id,
            change_source=ChangeSource.PIPELINE,
            status="proposed",
        ))

    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_to_delta(
    bundle: ExtractionBundle,
    provider: LLMProvider,
    namespace: str = DEFAULT_NAMESPACE,
    tbox: TBoxSummary | None = None,
) -> OntologyDelta:
    """
    Propose OWL triples for every entity in *bundle*.

    One LLM call is made per entity.  All returned entries have
    ``status="proposed"`` and ``change_source=ChangeSource.PIPELINE``.

    A failed mapping for one entity emits a warning and is skipped so the
    rest of the bundle is still processed.

    Args:
        bundle:    Output of :func:`~ontograph.ingest.extractor.extract`.
        provider:  Any :class:`~ontograph.llm.base.LLMProvider`.
        namespace: Base IRI namespace for subject and predicate IRIs.
        tbox:      Optional :class:`~ontograph.utils.owl.TBoxSummary` loaded
                   from a TBox OWL file.  When provided the LLM is instructed
                   to prefer existing class and property names.

    Returns:
        An :class:`~ontograph.models.ontology.OntologyDelta` with
        ``status="proposed"`` entries ready for human review.
    """
    all_entries: list[OntologyDeltaEntry] = []

    for entity in bundle.entities:
        messages = _build_mapping_messages(entity, namespace, tbox=tbox)
        request = LLMRequest(
            messages=messages,
            response_model=EntityMappingResponse,
            temperature=0.0,
            max_tokens=2048,
        )

        try:
            response = provider.complete(request)
        except Exception as exc:
            warnings.warn(
                f"Mapping failed for entity '{entity.text_span}' "
                f"(id={entity.id}): {exc}",
                stacklevel=2,
            )
            continue

        mapping: EntityMappingResponse = response.parsed
        all_entries.extend(_convert_mapping(mapping, entity, namespace))

    delta_id = hashlib.sha256(
        f"{bundle.id}:{namespace}".encode()
    ).hexdigest()[:16]

    return OntologyDelta(
        id=delta_id,
        extraction_bundle_id=bundle.id,
        base_ontology_iri=namespace,
        entries=all_entries,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
