"""
ontograph/reconstruction/runner.py — Orchestrate both reconstruction arms.

Two arms are compared:
  - ontograph: full pipeline (load → chunk → extract → map → align → working OWL)
  - direct:    single LLM call on raw document (no chunking, no section awareness)

Both arms are evaluated against the same source OWL (ground truth) using the
existing evaluator, then wrapped into a ReconstructionReport and saved.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ontograph.llm.base import LLMProvider
from ontograph.reconstruction.direct_extractor import extract_direct
from ontograph.reconstruction.schema import (
    ArmDebug,
    ArmResult,
    ReconstructionDebug,
    ReconstructionReport,
    TripleDetail,
)
from ontograph.utils.owl import (
    add_entries,
    copy_tbox,
    empty_graph,
    load_graph,
    owl_to_delta,
    read_tbox_summary,
    save_graph,
)
from ontograph.evaluator.comparator import evaluate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _eval_to_arm_result(arm: str, report) -> ArmResult:
    """Convert an EvaluationReport to an ArmResult."""
    return ArmResult(
        arm=arm,
        individual_precision=report.individuals.precision,
        individual_recall=report.individuals.recall,
        individual_f1=report.individuals.f1,
        triple_precision=report.triples.precision,
        triple_recall=report.triples.recall,
        triple_f1=report.triples.f1,
        triple_count_source=report.triples.source_count,
        triple_count_predicted=report.triples.working_count,
    )


def _synthesize_document(source_owl: Path, provider: LLMProvider) -> Path:
    """Synthesize a Markdown document from the source OWL and save it."""
    from ontograph.synthesizer import generate

    g     = load_graph(source_owl, fmt="xml")
    tbox  = read_tbox_summary(source_owl, fmt="xml")
    delta = owl_to_delta(g, delta_id=source_owl.stem, base_iri=tbox.namespace)

    title = " ".join(
        w.capitalize()
        for w in source_owl.stem.replace("-", " ").replace("_", " ").split()
    ) + " Design Description"

    doc = generate(delta, provider, title=title)

    # Strip citation anchors for cleaner document text
    clean = re.sub(r"\s*\[T-\d{3}\]", "", doc.markdown)
    clean = re.sub(r"  +", " ", clean)

    out_path = Path("data/raw") / f"{source_owl.stem}_synthesized.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(clean, encoding="utf-8")
    return out_path


def _run_ontograph_arm(
    source_owl: Path,
    document_path: Path,
    provider: LLMProvider,
    namespace: str,
    tbox,
    working_owl_path: Path,
) -> ArmResult:
    """
    Run the full onto-graph pipeline on *document_path* and evaluate the result.

    Steps: load → chunk → extract → map → align → auto-approve → write OWL → evaluate
    """
    from ontograph.ingest.loader import load_document
    from ontograph.ingest.chunker import chunk
    from ontograph.ingest.extractor import extract
    from ontograph.ingest.mapper import map_to_delta
    from ontograph.ingest.aligner import align, apply_decisions

    # Load and chunk
    raw = load_document(document_path)
    art = chunk(raw)

    # Extract entities (pass tbox so the prompt uses ontology class names, not hardcoded types)
    bundle = extract(art, provider, tbox=tbox)

    # Map to OWL delta
    delta = map_to_delta(bundle, provider, namespace=namespace, tbox=tbox)

    # Align (deduplicate entity mentions)
    alignment = align(bundle, delta, provider=provider)
    if alignment.approved_decisions():
        delta = apply_decisions(delta, alignment)

    # Auto-approve all proposed entries
    updated_entries = [
        e.model_copy(update={"status": "approved"}) if e.status == "proposed" else e
        for e in delta.entries
    ]
    delta = delta.model_copy(update={"entries": updated_entries})

    # Write working OWL (TBox + approved ABox triples)
    g = empty_graph()
    copy_tbox(source_owl, g, fmt="xml")
    add_entries(g, delta.approved_entries())

    # Cross-OWL IRI alignment: rename extracted individual IRIs to match source OWL locals
    # (e.g. "Vertiport_A" → "VertiportA", "Joby_S4" → "JobyS4").
    # Applied to the graph *before* saving so the OWL on disk, the debug file, and the
    # evaluated metrics all reflect the same aligned state.
    from ontograph.evaluator.comparator import _abox_individuals
    from ontograph.utils.iri_align import apply_iri_remap, cross_iri_align
    src_locals = [
        iri[len(namespace):]
        for iri in _abox_individuals(load_graph(source_owl, fmt="xml"))
        if iri.startswith(namespace)
    ]
    if src_locals:
        wrk_locals = [
            iri[len(namespace):]
            for iri in _abox_individuals(g)
            if iri.startswith(namespace)
        ]
        iri_mapping = cross_iri_align(wrk_locals, src_locals, provider)
        if iri_mapping:
            g = apply_iri_remap(g, iri_mapping, namespace)

    working_owl_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(g, working_owl_path, fmt="xml")

    # Evaluate against source OWL (no provider needed — alignment already applied above)
    report = evaluate(source_owl, working_owl_path, fmt="xml")
    return _eval_to_arm_result("ontograph", report)


def _run_direct_arm(
    source_owl: Path,
    document_path: Path,
    provider: LLMProvider,
    namespace: str,
    class_locals: list[str],
    property_locals: list[str],
    working_owl_path: Path,
) -> ArmResult:
    """
    Run the direct LLM arm: pour raw document + TBox vocabulary → single call → evaluate.

    The LLM receives class names and property names (same TBox the onto-graph arm uses)
    but NOT individual names — it must discover them from the document text.

    Steps: read document → extract_direct → convert to OWL graph → evaluate
    """
    from rdflib import Graph, Literal as RDFLiteral, Namespace
    from rdflib.namespace import OWL, RDF

    doc_text = document_path.read_text(encoding="utf-8")

    # Single LLM call — no chunking, no section tracking, no individual hints
    result = extract_direct(
        document_text=doc_text,
        namespace=namespace,
        class_locals=class_locals,
        property_locals=property_locals,
        provider=provider,
    )

    # Convert DirectExtractionResult → rdflib Graph
    NS = Namespace(namespace)
    g  = Graph()
    g.bind("ns", NS)

    for triple in result.triples:
        subj = NS[triple.subject]
        g.add((subj, RDF.type, OWL.NamedIndividual))
        if triple.rdf_type:
            g.add((subj, RDF.type, NS[triple.rdf_type]))   # class assertion
        g.add((subj, NS[triple.property], RDFLiteral(triple.value)))

    working_owl_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(g, working_owl_path, fmt="xml")

    # Evaluate against source OWL
    report = evaluate(source_owl, working_owl_path, fmt="xml")
    return _eval_to_arm_result("direct", report)


# ---------------------------------------------------------------------------
# Debug detail computation
# ---------------------------------------------------------------------------

def _compute_arm_debug(
    arm: str,
    source_owl: Path,
    working_owl: Path,
    fmt: str = "xml",
) -> ArmDebug:
    """
    Re-load both OWL graphs and compute the full matched / missed / extra sets
    for individuals and triples.  Called after evaluate() so no double LLM cost.
    """
    from ontograph.evaluator.comparator import _abox_individuals, _subject_triples

    src_g = load_graph(source_owl, fmt=fmt)
    wrk_g = load_graph(working_owl, fmt=fmt)

    src_inds = _abox_individuals(src_g)
    wrk_inds = _abox_individuals(wrk_g)

    matched_inds = sorted(src_inds & wrk_inds)
    missed_inds  = sorted(src_inds - wrk_inds)
    extra_inds   = sorted(wrk_inds - src_inds)

    src_triples = _subject_triples(src_g, src_inds)
    wrk_triples = _subject_triples(wrk_g, wrk_inds)

    def _to_detail(triple_set: set) -> list[TripleDetail]:
        return [
            TripleDetail(subject=s, predicate=p, object=o)
            for s, p, o in sorted(triple_set)
        ]

    return ArmDebug(
        arm=arm,
        matched_individuals=matched_inds,
        missed_individuals=missed_inds,
        extra_individuals=extra_inds,
        matched_triples=_to_detail(src_triples & wrk_triples),
        missed_triples=_to_detail(src_triples - wrk_triples),
        extra_triples=_to_detail(wrk_triples - src_triples),
    )


# ---------------------------------------------------------------------------
# Winner selection (extracted for testability)
# ---------------------------------------------------------------------------

def pick_winner(arms: list[ArmResult]) -> str | None:
    """Return the arm name with higher triple_f1, or None if tied / single arm."""
    if len(arms) == 2:
        if arms[0].triple_f1 > arms[1].triple_f1:
            return arms[0].arm
        elif arms[1].triple_f1 > arms[0].triple_f1:
            return arms[1].arm
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_reconstruction(
    source_owl: Path | str,
    document_path: Path | str | None = None,
    provider: LLMProvider | None = None,
    namespace: str | None = None,
    mode: Literal["ontograph", "direct", "both"] = "both",
    working_owl_path: Path | str | None = None,
    save_dir: Path | str | None = None,
) -> ReconstructionReport:
    """
    Orchestrate one or both reconstruction arms and return a ReconstructionReport.

    Args:
        source_owl:       Ground-truth OWL file (RDF/XML format).
        document_path:    Pre-synthesized document (Markdown/TXT).  If None,
                          the synthesizer is called to generate one from the OWL.
        provider:         LLM provider instance.
        namespace:        Ontology namespace IRI.  Auto-detected from source OWL
                          if not provided.
        mode:             Which arm(s) to run: "ontograph", "direct", or "both".
        working_owl_path: Output path for the onto-graph arm's working OWL.
                          Defaults to data/reconstruction/<stem>_ontograph.owl.
        save_dir:         Directory where the ReconstructionReport is saved.
                          Defaults to data/evaluations.

    Returns:
        A :class:`~ontograph.reconstruction.schema.ReconstructionReport` with
        scores for each arm and the winner (arm with higher triple F1).
    """
    source_owl = Path(source_owl)

    # Auto-detect namespace from source OWL
    tbox = read_tbox_summary(source_owl, fmt="xml")
    ns   = namespace or tbox.namespace

    # Synthesize document if not provided
    if document_path is None:
        if provider is None:
            raise ValueError("provider is required when document_path is None")
        document_path = _synthesize_document(source_owl, provider)
    document_path = Path(document_path)

    # TBox vocabulary for the direct arm (same info the onto-graph arm uses via map_to_delta)
    # Individual names are intentionally NOT passed — the LLM must discover them from text.
    class_locals    = sorted(tbox.classes)
    property_locals = sorted(
        {dp.local_name for dp in tbox.datatype_properties}
        | set(tbox.object_properties)
    )

    # Resolve working OWL paths
    recon_dir       = Path("data/reconstruction")
    ontograph_owl   = Path(working_owl_path) if working_owl_path else (
        recon_dir / f"{source_owl.stem}_ontograph.owl"
    )
    direct_owl      = recon_dir / f"{source_owl.stem}_direct.owl"

    # Run arm(s) — track which OWL paths were produced for debug computation
    arms: list[ArmResult] = []
    arm_owls: list[tuple[str, Path]] = []  # (arm_name, working_owl_path)

    if mode in ("ontograph", "both"):
        if provider is None:
            raise ValueError("provider is required for the ontograph arm")
        arms.append(_run_ontograph_arm(
            source_owl, document_path, provider, ns, tbox, ontograph_owl,
        ))
        arm_owls.append(("ontograph", ontograph_owl))

    if mode in ("direct", "both"):
        if provider is None:
            raise ValueError("provider is required for the direct arm")
        arms.append(_run_direct_arm(
            source_owl, document_path, provider, ns,
            class_locals, property_locals, direct_owl,
        ))
        arm_owls.append(("direct", direct_owl))

    # Determine winner (only meaningful when both arms ran)
    winner = pick_winner(arms)

    report_id = hashlib.sha256(
        f"{source_owl.resolve()}|{document_path.resolve()}|{mode}".encode()
    ).hexdigest()[:16]

    created_at = datetime.now(timezone.utc).isoformat()

    report = ReconstructionReport(
        id=report_id,
        created_at=created_at,
        source_owl_path=str(source_owl),
        document_path=str(document_path),
        provider=provider.provider_name if provider else "unknown",
        arms=arms,
        winner=winner,
    )

    # Save summary report
    from ontograph.utils.io import save
    out_dir = Path(save_dir) if save_dir else Path("data/evaluations")
    save(report, out_dir)

    # Compute and save per-arm debug detail (matched / missed / extra individuals + triples)
    arm_debug_list = [
        _compute_arm_debug(arm_name, source_owl, owl_path)
        for arm_name, owl_path in arm_owls
    ]
    debug = ReconstructionDebug(
        id=report_id,
        created_at=created_at,
        source_owl_path=str(source_owl),
        arms=arm_debug_list,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{report_id}_debug.json").write_text(
        debug.model_dump_json(indent=2), encoding="utf-8"
    )

    return report
