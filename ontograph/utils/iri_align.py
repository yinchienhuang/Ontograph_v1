"""
ontograph/utils/iri_align.py — Shared cross-OWL IRI alignment utilities.

Aligns individual local names between a working OWL and a source OWL when
the two graphs use different names for the same real-world entity
(e.g. "SerialPeripheralInterface" vs "SPI").

Detection strategy (in priority order):
  1. Exact / separator-normalised match    → score 1.0  (auto-approved)
  2. CamelCase-aware acronym check         → score 1.0  (auto-approved)
  3. Token Jaccard on CamelCase-split words
     - score >= IRI_AUTO_THRESHOLD         → auto-approved (no LLM)
     - score in [IRI_LLM_THRESHOLD, AUTO)  → LLM disambiguation (requires provider)
     - score < IRI_LLM_THRESHOLD           → discarded

Public API:
  iri_similarity(a, b)            → float
  cross_iri_align(working, source, provider) → {working_local: source_local}
  apply_iri_remap(g, mapping, ns) → Graph
  IriPairJudgment                 (Pydantic model — LLM response schema)
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ontograph.llm.base import LLMMessage, LLMProvider, LLMRequest


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

IRI_AUTO_THRESHOLD: float = 0.85   # auto-approve without LLM
IRI_LLM_THRESHOLD:  float = 0.40   # minimum score to attempt LLM disambiguation


# ---------------------------------------------------------------------------
# LLM response schema
# ---------------------------------------------------------------------------

class IriPairJudgment(BaseModel):
    """Structured output from the LLM for a single IRI pair."""
    same_entity: bool = Field(
        description="True if both local names name the same real-world entity"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the judgment",
    )


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def split_camel(s: str) -> str:
    """Split a CamelCase or underscore-separated IRI local name into spaced words.

    Examples::

        "SerialPeripheralInterface" → "Serial Peripheral Interface"
        "OBC_1"                     → "OBC 1"
        "powerW"                    → "power W"
    """
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)         # camelCase → camel Case
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)    # ABCDef → ABC Def
    s = re.sub(r"[_\-]+", " ", s)                       # snake_case / kebab-case
    return s


def iri_similarity(a: str, b: str) -> float:
    """Score similarity between two IRI local names (0–1).

    IRI locals are CamelCase; they are split into words before applying the
    acronym and Jaccard checks from :mod:`ontograph.ingest.aligner`.

    Checks (in order): exact → separator-normalised → acronym → token Jaccard.
    """
    from ontograph.ingest.aligner import _is_acronym, _token_jaccard, _strip_separators

    if a.lower() == b.lower():
        return 1.0
    if _strip_separators(a) == _strip_separators(b):
        return 1.0

    a_words = split_camel(a)
    b_words = split_camel(b)

    if _is_acronym(a, b_words) or _is_acronym(b, a_words):
        return 1.0

    return _token_jaccard(a_words, b_words)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def llm_iri_judge(
    working_local: str,
    source_local: str,
    provider: LLMProvider,
) -> IriPairJudgment:
    """Ask the LLM whether two IRI local names refer to the same real-world entity."""
    messages = [
        LLMMessage(
            role="system",
            content=(
                "You are an ontology entity resolution expert. "
                "Decide whether two OWL individual local names refer to the same real-world entity.\n"
                "Examples where they ARE the same: SPI / SerialPeripheralInterface, OBC / OnboardComputer.\n"
                "Return JSON with 'same_entity' (bool) and 'confidence' (0.0–1.0)."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"Source ontology individual: \"{source_local}\"\n"
                f"Reconstructed ontology individual: \"{working_local}\"\n\n"
                "Are these the same real-world entity?"
            ),
        ),
    ]
    request = LLMRequest(
        messages=messages,
        response_model=IriPairJudgment,
        temperature=0.0,
        max_tokens=128,
    )
    try:
        response = provider.complete(request)
        return response.parsed
    except Exception:
        return IriPairJudgment(same_entity=False, confidence=0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cross_iri_align(
    working_locals: list[str],
    source_locals: list[str],
    provider: LLMProvider | None,
) -> dict[str, str]:
    """Map working-OWL individual local names to source-OWL individual local names.

    For each working local, the best-scoring source local is found:

    - Score >= :data:`IRI_AUTO_THRESHOLD` → mapped immediately (no LLM needed)
    - :data:`IRI_LLM_THRESHOLD` <= score < :data:`IRI_AUTO_THRESHOLD` and *provider*
      available → LLM disambiguation
    - Otherwise → not mapped

    Args:
        working_locals: Individual local names from the working (reconstructed) OWL.
        source_locals:  Individual local names from the source (ground-truth) OWL.
        provider:       Optional LLM provider for ambiguous mid-range pairs.

    Returns:
        ``{working_local: source_local}`` for all confident matches where the names differ.
        Exact-match working locals (already in *source_locals*) are excluded.
    """
    source_set = set(source_locals)
    mapping: dict[str, str] = {}

    for w in working_locals:
        if w in source_set:
            continue  # already an exact match — no renaming needed

        best_score = 0.0
        best_src: str | None = None

        for s in source_locals:
            score = iri_similarity(w, s)
            if score > best_score:
                best_score = score
                best_src = s

        if best_src is None or best_score < IRI_LLM_THRESHOLD:
            continue

        if best_score >= IRI_AUTO_THRESHOLD:
            mapping[w] = best_src
        elif provider is not None:
            judgment = llm_iri_judge(w, best_src, provider)
            if judgment.same_entity and judgment.confidence >= IRI_AUTO_THRESHOLD:
                mapping[w] = best_src

    return mapping


def apply_iri_remap(g, mapping: dict[str, str], namespace: str):
    """Return a new rdflib Graph with individual IRIs renamed per *mapping*.

    Args:
        g:         Source rdflib Graph.
        mapping:   ``{old_local_name: new_local_name}`` renaming map.
        namespace: Ontology namespace string (e.g. ``"http://example.org/ns#"``).
                   Only URIRef nodes that start with this namespace are candidates.

    Returns:
        A new :class:`rdflib.Graph` with renamed IRIs.  If *mapping* is empty or
        all old == new, the original graph is returned unchanged.
    """
    from rdflib import Graph as RDFGraph, URIRef

    if not mapping:
        return g

    iri_map = {
        URIRef(namespace + old): URIRef(namespace + new)
        for old, new in mapping.items()
        if old != new
    }
    if not iri_map:
        return g

    new_g = RDFGraph()
    for prefix, uri in g.namespaces():
        new_g.bind(prefix, uri)

    for s, p, o in g:
        new_s = iri_map.get(s, s)
        new_o = iri_map.get(o, o) if isinstance(o, URIRef) else o
        new_g.add((new_s, p, new_o))

    return new_g
