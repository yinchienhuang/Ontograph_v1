"""
evaluator/comparator.py — Compare a working OWL against a source (ground-truth) OWL.

The two key metrics are:

  individual recall/precision/F1
      How many ABox named individuals from the source OWL were recovered
      (i.e. appear with the same IRI) in the working OWL?

  triple recall/precision/F1
      For the ABox individuals that exist in each graph, how many of their
      property assertions (s, p, o) triples match exactly?

TBox triples (class declarations, property axioms, subClassOf, etc.) are
automatically excluded — only triples whose *subject* is an ABox individual
are counted.  This means the comparison is independent of whether the TBox
was merged into the working OWL.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import OWL, RDF

from ontograph.evaluator.metrics import EvaluationReport, IndividualMetrics, TripleMetrics
from ontograph.llm.base import LLMProvider
from ontograph.utils.owl import _SCHEMA_TYPES, load_graph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_namespace(g: Graph) -> str | None:
    """Return the dominant namespace of named individuals in *g*, or None.

    Counts the namespace prefix shared by the most named individuals and returns
    it if it accounts for at least half of all individuals (avoids returning an
    irrelevant namespace when the graph mixes multiple ontologies).
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for s in g.subjects(RDF.type, OWL.NamedIndividual):
        uri = str(s)
        # Namespace is everything up to (and including) the last # or /
        for sep in ("#", "/"):
            idx = uri.rfind(sep)
            if idx != -1:
                counts[uri[: idx + 1]] += 1
                break

    if not counts:
        return None
    ns, top_count = counts.most_common(1)[0]
    total = sum(counts.values())
    return ns if top_count >= total / 2 else None


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0.0 else 0.0


def _abox_individuals(g: Graph) -> set[str]:
    """
    Return IRIs (as strings) of all ABox named individuals in *g*.

    Mirrors the logic used in owl_to_delta and step_compare so that results
    are consistent across the pipeline.
    """
    # Collect schema-axiom subjects (TBox) to exclude
    schema_subjects: set[URIRef] = set()
    for schema_type in _SCHEMA_TYPES:
        for s in g.subjects(RDF.type, schema_type):
            if not isinstance(s, BNode):
                schema_subjects.add(s)  # type: ignore[arg-type]

    inds: set[str] = set()

    # Explicit owl:NamedIndividual declarations
    for s in g.subjects(RDF.type, OWL.NamedIndividual):
        if not isinstance(s, BNode):
            inds.add(str(s))

    # Anything typed to a non-schema class
    for s, _p, o in g.triples((None, RDF.type, None)):
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue
        if s in schema_subjects:
            continue
        if o not in _SCHEMA_TYPES:
            inds.add(str(s))

    return inds


def _subject_triples(g: Graph, individuals: set[str]) -> set[tuple[str, str, str]]:
    """
    Return all non-blank-node triples from *g* whose subject is in *individuals*.

    Structural declarations (``rdf:type owl:NamedIndividual``) are excluded —
    they are present in every source OWL but never written by the pipeline,
    so including them would uniformly penalise recall without measuring knowledge
    reconstruction quality.

    Each triple is represented as a plain (subject, predicate, object) string
    tuple so set operations work correctly across graphs.
    """
    _NAMED_IND = str(OWL.NamedIndividual)

    result: set[tuple[str, str, str]] = set()
    for ind_str in individuals:
        uri = URIRef(ind_str)
        for s, p, o in g.triples((uri, None, None)):
            if isinstance(o, BNode):
                continue
            # Skip structural owl:NamedIndividual declarations
            if str(p) == str(RDF.type) and str(o) == _NAMED_IND:
                continue
            result.add((str(s), str(p), str(o)))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    source_owl: Path | str,
    working_owl: Path | str,
    fmt: str = "xml",
    provider: LLMProvider | None = None,
) -> EvaluationReport:
    """
    Compare a working OWL file against a source (ground-truth) OWL.

    Args:
        source_owl:  Path to the ground-truth OWL file (e.g. cubesatontology.owl).
        working_owl: Path to the reconstructed working OWL produced by the pipeline.
        fmt:         rdflib format string — ``"xml"`` for RDF/XML (default),
                     ``"turtle"`` for Turtle, etc.

    Returns:
        :class:`~ontograph.evaluator.metrics.EvaluationReport` with both
        individual-level and triple-level precision/recall/F1 scores.
    """
    source_owl  = Path(source_owl)
    working_owl = Path(working_owl)

    src_g = load_graph(source_owl,  fmt=fmt)
    wrk_g = load_graph(working_owl, fmt=fmt)

    # ── Optional cross-IRI alignment ──────────────────────────────────────
    # When a provider is supplied, attempt to rename working individuals to
    # match source IRI local names (e.g. "SerialPeripheralInterface" → "SPI").
    # With provider=None (the default) this block is skipped entirely, so
    # existing callers see no behaviour change.
    if provider is not None:
        namespace = _detect_namespace(src_g)
        if namespace:
            from ontograph.utils.iri_align import apply_iri_remap, cross_iri_align
            # Use _abox_individuals (class-assertion aware) rather than looking for
            # owl:NamedIndividual declarations — the pipeline never writes those.
            src_locals = [
                iri[len(namespace):]
                for iri in _abox_individuals(src_g)
                if iri.startswith(namespace)
            ]
            wrk_locals = [
                iri[len(namespace):]
                for iri in _abox_individuals(wrk_g)
                if iri.startswith(namespace)
            ]
            mapping = cross_iri_align(wrk_locals, src_locals, provider)
            if mapping:
                wrk_g = apply_iri_remap(wrk_g, mapping, namespace)

    # ── Individual-level metrics ───────────────────────────────────────────
    src_inds       = _abox_individuals(src_g)
    wrk_inds       = _abox_individuals(wrk_g)
    recovered_inds = src_inds & wrk_inds
    missed_inds    = src_inds - wrk_inds
    extra_inds     = wrk_inds - src_inds

    ind_recall    = len(recovered_inds) / len(src_inds) if src_inds    else 0.0
    ind_precision = len(recovered_inds) / len(wrk_inds) if wrk_inds    else 0.0
    ind_f1        = _f1(ind_precision, ind_recall)

    # ── Triple-level metrics (ABox triples only) ───────────────────────────
    src_triples       = _subject_triples(src_g, src_inds)
    wrk_triples       = _subject_triples(wrk_g, wrk_inds)
    recovered_triples = src_triples & wrk_triples
    missed_triples    = src_triples - wrk_triples
    extra_triples     = wrk_triples - src_triples

    tri_recall    = len(recovered_triples) / len(src_triples) if src_triples else 0.0
    tri_precision = len(recovered_triples) / len(wrk_triples) if wrk_triples else 0.0
    tri_f1        = _f1(tri_precision, tri_recall)

    report_id = hashlib.sha256(
        f"{source_owl.resolve()}|{working_owl.resolve()}".encode()
    ).hexdigest()[:16]

    return EvaluationReport(
        id=report_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_owl_path=str(source_owl),
        working_owl_path=str(working_owl),
        individuals=IndividualMetrics(
            source_count=len(src_inds),
            working_count=len(wrk_inds),
            recovered_count=len(recovered_inds),
            missed_count=len(missed_inds),
            extra_count=len(extra_inds),
            recall=ind_recall,
            precision=ind_precision,
            f1=ind_f1,
        ),
        triples=TripleMetrics(
            source_count=len(src_triples),
            working_count=len(wrk_triples),
            recovered_count=len(recovered_triples),
            missed_count=len(missed_triples),
            extra_count=len(extra_triples),
            recall=tri_recall,
            precision=tri_precision,
            f1=tri_f1,
        ),
        missed_individuals=sorted(missed_inds),
        extra_individuals=sorted(extra_inds),
    )
