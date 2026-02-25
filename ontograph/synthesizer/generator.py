"""
synthesizer/generator.py — Grounded document generation from OntologyDelta.

Takes the approved triples from an OntologyDelta and generates a structured
natural-language report (Markdown) where every factual paragraph is tied back
to the exact triples it encodes via inline citations like [T-001].

Pipeline:
    OntologyDelta (approved entries)
        → group by subject IRI
        → one LLM call per subject group  → SectionDraft
        → assemble full Markdown + ParagraphProvenance
        → SynthesizedDocument  (self_check=None, run separately)
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterator

from pydantic import BaseModel, Field

from ontograph.llm.base import LLMMessage, LLMProvider, LLMRequest
from ontograph.models.ontology import OntologyDelta, OntologyDeltaEntry
from ontograph.models.synthesis import ParagraphProvenance, SynthesizedDocument


# ---------------------------------------------------------------------------
# LLM response schemas (structured output contracts for the LLM)
# ---------------------------------------------------------------------------

class ParagraphDraft(BaseModel):
    """One paragraph returned by the LLM — text + the triples it encodes."""

    text: str = Field(
        description=(
            "One Markdown paragraph. Every factual claim MUST include its "
            "citation anchor, e.g. 'The thruster has a mass of 89 kg [T-001].'"
        )
    )
    cited_anchors: list[str] = Field(
        description="All citation anchors used in this paragraph, e.g. ['[T-001]', '[T-003]']"
    )


class SectionDraft(BaseModel):
    """One document section returned by the LLM for a single subject."""

    section_title: str = Field(
        description="A concise title for this section (no # prefix)"
    )
    paragraphs: list[ParagraphDraft] = Field(
        description="Ordered paragraphs. Must collectively cite every provided triple."
    )


# ---------------------------------------------------------------------------
# IRI → human-readable label
# ---------------------------------------------------------------------------

def label_from_iri(iri: str) -> str:
    """
    Extract a human-readable label from an IRI or prefixed name.

    Examples:
        'aero:ThrusterModule_42'          → 'Thruster Module 42'
        'http://example.org/aero#FuelTank' → 'Fuel Tank'
        'rdf:type'                         → 'type'
    """
    # Get local name after # or last /  or last :
    if "#" in iri:
        local = iri.split("#")[-1]
    elif "/" in iri:
        local = iri.rstrip("/").split("/")[-1]
    elif ":" in iri:
        local = iri.split(":")[-1]
    else:
        local = iri

    # CamelCase → spaces
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local)
    # Underscores → spaces
    label = label.replace("_", " ")
    # Collapse multiple spaces
    label = re.sub(r"\s+", " ", label).strip()
    return label


# ---------------------------------------------------------------------------
# Anchor map: entry_id → "[T-NNN]"
# ---------------------------------------------------------------------------

def build_anchor_map(entries: list[OntologyDeltaEntry]) -> dict[str, str]:
    """
    Build a stable mapping from OntologyDeltaEntry.id → citation anchor.

    Anchors are sequential [T-001], [T-002], … ordered by entry id.
    Stable ordering ensures the same delta always produces the same anchors.
    """
    sorted_ids = sorted(e.id for e in entries)
    return {eid: f"[T-{i+1:03d}]" for i, eid in enumerate(sorted_ids)}


def anchor_to_entry_id(anchor_map: dict[str, str]) -> dict[str, str]:
    """Reverse of build_anchor_map: '[T-001]' → entry_id."""
    return {v: k for k, v in anchor_map.items()}


# ---------------------------------------------------------------------------
# Triple formatting for LLM context
# ---------------------------------------------------------------------------

def _format_triple_for_llm(
    entry: OntologyDeltaEntry,
    anchor: str,
) -> str:
    """
    Format one triple as a single line for LLM context injection.

    Example:
        [T-001] ThrusterModule 42  hasMass  "89" (xsd:float)
    """
    s = label_from_iri(entry.triple.subject)
    p = label_from_iri(entry.triple.predicate)
    o = entry.triple.object

    if entry.triple.datatype:
        short_dt = entry.triple.datatype.split(":")[-1].split("#")[-1]
        o_str = f'"{o}" ({short_dt})'
    elif o.startswith("http") or ":" in o:
        o_str = label_from_iri(o)
    else:
        o_str = f'"{o}"'

    return f"{anchor}  {s}  —  {p}  →  {o_str}"


# ---------------------------------------------------------------------------
# Subject grouping
# ---------------------------------------------------------------------------

def _group_by_subject(
    entries: list[OntologyDeltaEntry],
) -> dict[str, list[OntologyDeltaEntry]]:
    """Group entries by their triple subject IRI, preserving insertion order."""
    groups: dict[str, list[OntologyDeltaEntry]] = defaultdict(list)
    for entry in entries:
        groups[entry.triple.subject].append(entry)
    return dict(groups)


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior aerospace systems engineer writing a formal design document.

STRICT RULES — violations will invalidate the document:
1. Use ONLY the facts provided in the triple list below. Do NOT add, infer, or \
assume any information not explicitly stated in a triple.
2. Every factual claim in your text MUST include its citation anchor (e.g., [T-001]) \
immediately after the claim.
3. Every triple provided MUST be cited at least once somewhere in your output.
4. Write in clear, professional aerospace engineering language (present tense, \
active voice where appropriate).
5. Return ONLY valid JSON matching the required schema. No preamble, no explanation.
"""


def _build_section_messages(
    subject_iri: str,
    entries: list[OntologyDeltaEntry],
    anchor_map: dict[str, str],
) -> list[LLMMessage]:
    subject_label = label_from_iri(subject_iri)

    triple_lines = "\n".join(
        _format_triple_for_llm(e, anchor_map[e.id])
        for e in entries
    )

    user_content = (
        f"Subject component: {subject_label}\n\n"
        f"Triples to encode (cite EVERY one):\n{triple_lines}\n\n"
        f"Write a document section about '{subject_label}' encoding ALL of the above triples. "
        f"Each paragraph in your output must include the relevant citation anchors inline."
    )

    return [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user",   content=user_content),
    ]


# ---------------------------------------------------------------------------
# Provenance extraction
# ---------------------------------------------------------------------------

def _anchors_in_text(text: str) -> list[str]:
    """Extract all [T-NNN] anchors from a Markdown string."""
    return re.findall(r"\[T-\d{3}\]", text)


def _build_provenance(
    sections: list[tuple[str, SectionDraft]],  # (subject_iri, draft)
    anchor_to_id: dict[str, str],
    para_offset: int = 0,
) -> list[ParagraphProvenance]:
    """
    Convert SectionDraft paragraphs into ParagraphProvenance entries.

    Anchors found in the text are resolved to entry IDs via anchor_to_id.
    Anchors in `cited_anchors` that aren't in the text are still included
    (the LLM may omit inline anchors while listing them in the field).
    """
    provenance: list[ParagraphProvenance] = []
    idx = para_offset

    for _subject_iri, draft in sections:
        for para in draft.paragraphs:
            # Collect from inline text + from the cited_anchors field
            inline_anchors = _anchors_in_text(para.text)
            all_anchors = sorted(set(inline_anchors) | set(para.cited_anchors))

            triple_ids = [
                anchor_to_id[a]
                for a in all_anchors
                if a in anchor_to_id
            ]
            provenance.append(ParagraphProvenance(
                paragraph_index=idx,
                triple_ids=triple_ids,
                citation_anchors=all_anchors,
            ))
            idx += 1

    return provenance


# ---------------------------------------------------------------------------
# Markdown assembly
# ---------------------------------------------------------------------------

def _assemble_markdown(
    title: str,
    sections: list[tuple[str, SectionDraft]],
) -> str:
    lines: list[str] = [f"# {title}", ""]

    for _subject_iri, draft in sections:
        lines.append(f"## {draft.section_title}")
        lines.append("")
        for para in draft.paragraphs:
            lines.append(para.text)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    delta: OntologyDelta,
    provider: LLMProvider,
    title: str = "System Design Document",
) -> SynthesizedDocument:
    """
    Generate a grounded Markdown design document from approved OntologyDelta entries.

    Each approved entry must be cited at least once in the output.
    Returns a SynthesizedDocument with full ParagraphProvenance.
    `self_check` is left as None — run `self_check.run_self_check()` separately.

    Raises:
        ValueError: if there are no approved entries in the delta.
    """
    approved = delta.approved_entries()
    if not approved:
        raise ValueError(
            f"OntologyDelta '{delta.id}' has no approved entries to synthesize."
        )

    # Build anchor map over ALL approved entries (global, not per-section)
    anchor_map = build_anchor_map(approved)
    anchor_to_id = anchor_to_entry_id(anchor_map)

    groups = _group_by_subject(approved)

    sections: list[tuple[str, SectionDraft]] = []
    for subject_iri, entries in groups.items():
        messages = _build_section_messages(subject_iri, entries, anchor_map)
        request = LLMRequest(
            messages=messages,
            response_model=SectionDraft,
            temperature=0.2,
            max_tokens=2048,
        )
        response = provider.complete(request)
        section_draft: SectionDraft = response.parsed
        sections.append((subject_iri, section_draft))

    markdown = _assemble_markdown(title, sections)
    provenance = _build_provenance(sections, anchor_to_id)

    doc_id = hashlib.sha256(
        f"{delta.id}:{title}".encode()
    ).hexdigest()[:16]

    return SynthesizedDocument(
        id=doc_id,
        ontology_delta_id=delta.id,
        title=title,
        markdown=markdown,
        provenance=provenance,
        self_check=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
