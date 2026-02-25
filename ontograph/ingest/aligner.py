"""
ingest/aligner.py — Entity deduplication for the OntologyDelta.

Detects that two extracted entity surface forms refer to the same real-world
concept (e.g. "FAA" and "Federal Aviation Administration") and proposes that
they share a single canonical OWL IRI.

Detection methods (in priority order):
  ACRONYM            — one string is an initialism of the other
  STRING_SIMILARITY  — token Jaccard overlap above a threshold
  LLM                — LLM judgment for ambiguous mid-range pairs

After decisions are approved, :func:`apply_decisions` rewrites the
``OntologyDelta`` so all merged entities point to the canonical IRI and
``skos:altLabel`` triples are inserted for every alias surface form.

Pipeline position:
    ExtractionBundle + OntologyDelta
        → :func:`align`
        → OntologyAlignmentBundle   (candidates + decisions)
        → :func:`apply_decisions`
        → aligned OntologyDelta     (IRIs rewritten, skos:altLabel inserted)
"""

from __future__ import annotations

import hashlib
import re
import warnings
from datetime import datetime, timezone
from itertools import combinations

from pydantic import BaseModel, Field

from ontograph.llm.base import LLMMessage, LLMProvider, LLMRequest
from ontograph.models.alignment import (
    AlignmentCandidate,
    AlignmentDecision,
    AlignmentMethod,
    OntologyAlignmentBundle,
)
from ontograph.models.extraction import ExtractedEntity, ExtractionBundle
from ontograph.models.ontology import (
    ChangeSource,
    OntologyDelta,
    OntologyDeltaEntry,
    OntologyTriple,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKOS_ALT_LABEL = "http://www.w3.org/2004/02/skos/core#altLabel"

DEFAULT_NAMESPACE = "http://example.org/aerospace#"

# Candidates with score >= this are auto-approved without an LLM call
_AUTO_APPROVE_THRESHOLD: float = 0.85

# Candidates with score in [_LLM_THRESHOLD, _AUTO_APPROVE_THRESHOLD)
# are sent to the LLM for a judgment call
_LLM_THRESHOLD: float = 0.40

# Minimum Jaccard similarity to even generate a candidate
_JACCARD_MIN: float = 0.40


# ---------------------------------------------------------------------------
# LLM response schema
# ---------------------------------------------------------------------------

class AlignmentJudgment(BaseModel):
    """Structured output from the LLM for one entity pair."""

    same_entity: bool = Field(
        description="True if both surface forms refer to the same real-world concept"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that this judgment is correct (0 = uncertain, 1 = certain)",
    )
    rationale: str = Field(description="One-sentence justification")
    canonical_surface: str = Field(
        description=(
            "The more formal / complete surface form to use as the canonical name. "
            "Must be exactly one of the two input surfaces."
        )
    )


_ALIGNMENT_SYSTEM = """\
You are an aerospace entity resolution expert.

Decide whether two entity surface forms refer to the same real-world concept
(e.g., "FAA" and "Federal Aviation Administration" are the same;
"thruster assembly" and "propellant tank" are different).

RULES:
  - same_entity = true only when you are confident they name the same thing.
  - canonical_surface must be exactly one of the two inputs — prefer the
    longer, more formal name (e.g., prefer full name over acronym).
  - confidence: 1.0 = certain, 0.5 = unsure.

Return ONLY valid JSON matching the schema. No preamble.
"""


def _build_alignment_messages(
    entity_a: ExtractedEntity,
    entity_b: ExtractedEntity,
) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=_ALIGNMENT_SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Entity A: \"{entity_a.text_span}\" (type: {entity_a.entity_type})\n"
                f"Entity B: \"{entity_b.text_span}\" (type: {entity_b.entity_type})\n\n"
                "Are these the same real-world concept?"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# String comparison helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "and", "or", "for", "in", "on",
    "at", "to", "by", "with", "from", "as",
})


def _is_acronym(short: str, long: str) -> bool:
    """Return True if *short* is an initialism of *long*.

    Articles and prepositions (``_STOP_WORDS``) are skipped when building
    initials, matching common conventions:

    - ``_is_acronym("FAA", "Federal Aviation Administration")`` → True
    - ``_is_acronym("NASA", "National Aeronautics and Space Administration")`` → True
    """
    if len(short) < 2 or len(short) > 8:
        return False
    words = re.findall(r"[A-Za-z]+", long)
    # Drop stop words — acronyms typically omit articles/prepositions
    content = [w for w in words if w.lower() not in _STOP_WORDS]
    if len(content) < len(short):
        return False
    initials = "".join(w[0].upper() for w in content)
    return short.upper() == initials


def _token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings (case-insensitive)."""
    ta = set(re.findall(r"[A-Za-z0-9]+", a.lower()))
    tb = set(re.findall(r"[A-Za-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _strip_separators(s: str) -> str:
    """Remove underscores, hyphens, and whitespace for normalised comparison."""
    return re.sub(r"[_\-\s]+", "", s).lower()


def _similarity_score(a: ExtractedEntity, b: ExtractedEntity) -> tuple[float, AlignmentMethod]:
    """Return (score, method) for the best match between two entity surface forms."""
    sa, sb = a.text_span, b.text_span

    # 1. Exact match (case-insensitive)
    if sa.strip().lower() == sb.strip().lower():
        return 1.0, AlignmentMethod.STRING_SIMILARITY

    # 2. Separator-normalised match — catches EPS1/EPS_1, COMMS1/COMMS_1
    if _strip_separators(sa) == _strip_separators(sb):
        return 1.0, AlignmentMethod.STRING_SIMILARITY

    # 3. Acronym check (either direction)
    if _is_acronym(sa, sb) or _is_acronym(sb, sa):
        return 1.0, AlignmentMethod.ACRONYM

    # 4. Token Jaccard
    jaccard = _token_jaccard(sa, sb)
    return jaccard, AlignmentMethod.STRING_SIMILARITY


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _candidate_id(id_a: str, id_b: str) -> str:
    """Stable sha256-based ID for a pair, order-independent."""
    key = "|".join(sorted([id_a, id_b]))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _pick_canonical(entity_a: ExtractedEntity, entity_b: ExtractedEntity) -> tuple[ExtractedEntity, ExtractedEntity]:
    """Return (canonical, alias) — prefer the longer, more formal surface form."""
    if len(entity_a.text_span) >= len(entity_b.text_span):
        return entity_a, entity_b
    return entity_b, entity_a


def _generate_candidates(entities: list[ExtractedEntity]) -> list[tuple[AlignmentCandidate, float]]:
    """
    Compare all pairs and return (candidate, score) for those above _JACCARD_MIN.

    Returned list is sorted by score descending.
    """
    results: list[tuple[AlignmentCandidate, float]] = []
    seen: set[frozenset[str]] = set()

    for ea, eb in combinations(entities, 2):
        pair_key = frozenset({ea.id, eb.id})
        if pair_key in seen:
            continue
        seen.add(pair_key)

        score, method = _similarity_score(ea, eb)
        if score < _JACCARD_MIN:
            continue

        cid = _candidate_id(ea.id, eb.id)
        rationale = (
            f"Acronym match" if method == AlignmentMethod.ACRONYM
            else f"Token Jaccard similarity = {score:.2f}"
        )
        results.append((
            AlignmentCandidate(
                id=cid,
                entity_id_a=ea.id,
                entity_id_b=eb.id,
                surface_a=ea.text_span,
                surface_b=eb.text_span,
                similarity_score=round(score, 4),
                method=method,
                rationale=rationale,
            ),
            score,
        ))

    results.sort(key=lambda t: t[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------

def _make_decision(
    candidate: AlignmentCandidate,
    entity_a: ExtractedEntity,
    entity_b: ExtractedEntity,
    status: str,
    canonical_entity_id: str | None = None,
) -> AlignmentDecision:
    canonical, alias = _pick_canonical(entity_a, entity_b)
    chosen_canonical = canonical_entity_id or canonical.id
    merged = [ea.id for ea in (entity_a, entity_b) if ea.id != chosen_canonical]
    return AlignmentDecision(
        candidate_id=candidate.id,
        canonical_entity_id=chosen_canonical,
        merged_entity_ids=merged,
        aliases=[entity_a.text_span, entity_b.text_span],
        status=status,
    )


# ---------------------------------------------------------------------------
# Entity IRI map
# ---------------------------------------------------------------------------

def _entity_iri_map(delta: OntologyDelta) -> dict[str, str]:
    """Map ``ExtractedEntity.id → triple.subject`` IRI from the delta entries."""
    result: dict[str, str] = {}
    for entry in delta.entries:
        eid = entry.source_entity_id
        if eid and eid not in result:
            result[eid] = entry.triple.subject
    return result


def _entry_id(subject: str, predicate: str, obj: str) -> str:
    raw = f"{subject}|{predicate}|{obj}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API — align()
# ---------------------------------------------------------------------------

def align(
    bundle: ExtractionBundle,
    delta: OntologyDelta,
    provider: LLMProvider | None = None,
    auto_approve_threshold: float = _AUTO_APPROVE_THRESHOLD,
    llm_threshold: float = _LLM_THRESHOLD,
) -> OntologyAlignmentBundle:
    """
    Detect duplicate entity mentions across an ``ExtractionBundle``.

    For each pair of entities that look similar:
    - Score >= *auto_approve_threshold* → decision is auto-approved
    - *llm_threshold* <= score < *auto_approve_threshold* → LLM disambiguation
      (requires *provider*; if absent the candidate is left ``"proposed"``)
    - Score < *llm_threshold* → pair is discarded

    Args:
        bundle:                 Output of :func:`~ontograph.ingest.extractor.extract`.
        delta:                  ``OntologyDelta`` produced by the mapper; used
                                to resolve entity IDs to OWL IRIs.
        provider:               Optional LLM provider for ambiguous pairs.
        auto_approve_threshold: Similarity score above which the decision is
                                automatically approved without LLM.
        llm_threshold:          Minimum similarity to trigger an LLM call.

    Returns:
        An :class:`~ontograph.models.alignment.OntologyAlignmentBundle`.
    """
    entities = bundle.entities
    id_to_entity: dict[str, ExtractedEntity] = {e.id: e for e in entities}

    # ── Generate candidates ──────────────────────────────────────────────
    candidates_with_scores = _generate_candidates(entities)
    candidates = [c for c, _ in candidates_with_scores]
    scores     = {c.id: s for c, s in candidates_with_scores}

    # ── Make decisions ───────────────────────────────────────────────────
    decisions: list[AlignmentDecision] = []

    for candidate in candidates:
        ea = id_to_entity[candidate.entity_id_a]
        eb = id_to_entity[candidate.entity_id_b]
        score = scores[candidate.id]

        if score >= auto_approve_threshold:
            # Auto-approve: high confidence, no LLM needed
            decisions.append(_make_decision(candidate, ea, eb, status="approved"))
            continue

        # Ambiguous — try LLM if available
        if provider is None:
            decisions.append(_make_decision(candidate, ea, eb, status="proposed"))
            continue

        messages = _build_alignment_messages(ea, eb)
        request = LLMRequest(
            messages=messages,
            response_model=AlignmentJudgment,
            temperature=0.0,
            max_tokens=256,
        )
        try:
            response = provider.complete(request)
        except Exception as exc:
            warnings.warn(
                f"LLM alignment call failed for pair "
                f"('{ea.text_span}', '{eb.text_span}'): {exc}",
                stacklevel=2,
            )
            decisions.append(_make_decision(candidate, ea, eb, status="proposed"))
            continue

        judgment: AlignmentJudgment = response.parsed
        if not judgment.same_entity:
            decisions.append(AlignmentDecision(
                candidate_id=candidate.id,
                canonical_entity_id=ea.id,
                merged_entity_ids=[],
                aliases=[],
                status="rejected",
                reviewer_note=f"LLM: {judgment.rationale}",
            ))
            continue

        # LLM says same entity — set canonical based on LLM's canonical_surface
        canonical_entity_id = (
            ea.id if ea.text_span == judgment.canonical_surface else eb.id
        )
        status = "approved" if judgment.confidence >= auto_approve_threshold else "proposed"
        decisions.append(_make_decision(
            candidate, ea, eb,
            status=status,
            canonical_entity_id=canonical_entity_id,
        ))

    # ── Build bundle ─────────────────────────────────────────────────────
    bundle_id = hashlib.sha256(
        f"{delta.id}:alignment".encode()
    ).hexdigest()[:16]

    return OntologyAlignmentBundle(
        id=bundle_id,
        ontology_delta_id=delta.id,
        candidates=candidates,
        decisions=decisions,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Public API — apply_decisions()
# ---------------------------------------------------------------------------

def apply_decisions(
    delta: OntologyDelta,
    alignment: OntologyAlignmentBundle,
) -> OntologyDelta:
    """
    Rewrite *delta* according to approved alignment decisions.

    For each approved decision:
    - All ``OntologyDeltaEntry`` objects whose ``source_entity_id`` is in
      ``decision.merged_entity_ids`` have their ``triple.subject`` rewritten
      to the canonical entity's IRI.
    - A ``skos:altLabel`` triple is appended to the delta for every alias
      surface form, with ``status="approved"`` and
      ``change_source=ChangeSource.ALIGNMENT``.

    Entries whose subject IRI cannot be resolved (entity never mapped) are
    left unchanged and a warning is emitted.

    Args:
        delta:     The ``OntologyDelta`` to rewrite (not modified in place).
        alignment: The ``OntologyAlignmentBundle`` returned by :func:`align`.

    Returns:
        A new ``OntologyDelta`` with rewritten entries.
    """
    approved = alignment.approved_decisions()
    if not approved:
        return delta

    iri_map = _entity_iri_map(delta)  # entity_id → subject IRI

    # Build rewrite map: merged IRI → canonical IRI
    rewrite: dict[str, str] = {}
    alt_labels: list[tuple[str, str]] = []  # (canonical_iri, surface)

    for decision in approved:
        canonical_iri = iri_map.get(decision.canonical_entity_id)
        if canonical_iri is None:
            warnings.warn(
                f"Canonical entity '{decision.canonical_entity_id}' not found "
                "in delta IRI map — skipping decision.",
                stacklevel=2,
            )
            continue

        for mid in decision.merged_entity_ids:
            merged_iri = iri_map.get(mid)
            if merged_iri is None:
                continue
            if merged_iri != canonical_iri:
                rewrite[merged_iri] = canonical_iri

        for alias in decision.aliases:
            alt_labels.append((canonical_iri, alias))

    # Rewrite entries
    new_entries: list[OntologyDeltaEntry] = []
    for entry in delta.entries:
        subj = entry.triple.subject
        if subj in rewrite:
            canon = rewrite[subj]
            new_triple = entry.triple.model_copy(update={"subject": canon})
            new_id = _entry_id(canon, entry.triple.predicate, entry.triple.object)
            new_entries.append(entry.model_copy(update={
                "id": new_id,
                "triple": new_triple,
                "change_source": ChangeSource.ALIGNMENT,
            }))
        else:
            new_entries.append(entry)

    # Deduplicate (same triple might appear after rewriting)
    seen_ids: set[str] = set()
    deduped: list[OntologyDeltaEntry] = []
    for e in new_entries:
        if e.id not in seen_ids:
            seen_ids.add(e.id)
            deduped.append(e)

    # Append skos:altLabel entries
    seen_alt: set[tuple[str, str]] = set()
    for canonical_iri, surface in alt_labels:
        key = (canonical_iri, surface)
        if key in seen_alt:
            continue
        seen_alt.add(key)
        triple = OntologyTriple(
            subject=canonical_iri,
            predicate=_SKOS_ALT_LABEL,
            object=surface,
            language="en",
        )
        alt_entry = OntologyDeltaEntry(
            id=_entry_id(canonical_iri, _SKOS_ALT_LABEL, surface),
            triple=triple,
            rationale=f"Alias surface form for canonical entity",
            confidence=1.0,
            change_source=ChangeSource.ALIGNMENT,
            status="approved",
        )
        if alt_entry.id not in seen_ids:
            seen_ids.add(alt_entry.id)
            deduped.append(alt_entry)

    return delta.model_copy(update={"entries": deduped})
