"""
scripts/run_pipeline.py - End-to-end Ontograph pipeline runner.

Runs every stage automatically.  Use --auto-approve to skip the interactive
human-review step; without it the pipeline stops after alignment and tells
you which file to review.

Two modes
---------
Document mode (--input):
    PDF / TXT / MD -> chunk -> extract -> map -> align -> [review] -> working.owl

Validation-loop mode (--from-owl):
    OWL individuals -> synthesize text -> chunk -> extract -> map ->
    align -> [review] -> working.owl  +  comparison report

Usage
-----
    # Document -> ontology, human review required:
    python scripts/run_pipeline.py \\
        --input    data/raw/report.pdf \\
        --tbox     data/ontology/cubesatontology.owl \\
        --owl      data/ontology/working.owl \\
        --provider claude

    # Same, but skip human review (auto-approve all proposed triples):
    python scripts/run_pipeline.py \\
        --input    data/raw/report.pdf \\
        --tbox     data/ontology/cubesatontology.owl \\
        --owl      data/ontology/working.owl \\
        --provider claude \\
        --auto-approve

    # OWL validation loop (synthesize -> ingest -> compare):
    python scripts/run_pipeline.py \\
        --from-owl data/ontology/cubesatontology.owl \\
        --owl      data/ontology/working.owl \\
        --provider claude \\
        --auto-approve

    # Cheap test (limit to first 5 chunks):
    python scripts/run_pipeline.py \\
        --input data/raw/report.pdf --tbox ... --owl ... \\
        --provider claude --auto-approve --max-chunks 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_api_key(provider_name: str) -> None:
    _key_map = {
        "claude": ("ANTHROPIC_API_KEY",  "https://console.anthropic.com/"),
        "gpt-4o": ("OPENAI_API_KEY",     "https://platform.openai.com/api-keys"),
        "gemini": ("GOOGLE_API_KEY",     "https://aistudio.google.com/app/apikey"),
    }
    if provider_name in _key_map:
        env_var, url = _key_map[provider_name]
        if not os.getenv(env_var):
            console.print(
                f"\n[red bold]Missing {env_var}[/red bold]\n"
                f"  Set it in [cyan].env[/cyan] — get a key at {url}\n"
            )
            sys.exit(1)


def _rel(path: Path) -> str:
    """Return path relative to ROOT for compact display."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _section(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold yellow]{title}[/bold yellow]", style="yellow"))


def _generated(label: str, path: Path) -> None:
    console.print(f"  [dim]-> {label}:[/dim] [cyan]{_rel(path)}[/cyan]")


# ---------------------------------------------------------------------------
# Step 0 — Synthesize text from OWL (--from-owl mode only)
# ---------------------------------------------------------------------------

def step_synthesize(
    owl_path: Path,
    provider,
) -> Path:
    """Generate synthetic Markdown from OWL ABox individuals."""
    from ontograph.utils import owl as owl_utils
    from ontograph.synthesizer import generate

    _section("Step 0 — Synthesize text from OWL")

    g     = owl_utils.load_graph(owl_path, fmt="xml")
    tbox  = owl_utils.read_tbox_summary(owl_path, fmt="xml")
    delta = owl_utils.owl_to_delta(g, delta_id=owl_path.stem, base_iri=tbox.namespace)

    approved = delta.approved_entries()
    n_subjects = len({e.triple.subject for e in approved})
    console.print(
        f"  Individuals  : [cyan]{n_subjects}[/cyan]\n"
        f"  ABox triples : [cyan]{len(approved)}[/cyan]"
    )

    if not approved:
        console.print("[red]No ABox individuals found in OWL — cannot synthesize.[/red]")
        sys.exit(1)

    title = " ".join(
        w.capitalize()
        for w in owl_path.stem.replace("-", " ").replace("_", " ").split()
    ) + " Design Description"

    console.print(f"\n[bold]Generating document:[/bold] {title}")
    doc = generate(delta, provider, title=title)
    console.print(
        f"  [green]✓ Generated[/green] — "
        f"{len(doc.provenance)} paragraphs, "
        f"{len(doc.triples_cited())} triples cited"
    )

    # Strip [T-NNN] citation anchors for cleaner text
    clean = re.sub(r"\s*\[T-\d{3}\]", "", doc.markdown)
    clean = re.sub(r"  +", " ", clean)

    out_path = DATA / "raw" / f"{owl_path.stem}_synthesized.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(clean, encoding="utf-8")

    _generated("Synthesized document", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Step 1 — Convert + Chunk
# ---------------------------------------------------------------------------

def step_ingest(doc_path: Path, max_chunks: int | None) -> tuple:
    """Convert document -> DocumentArtifact."""
    from ontograph.ingest.loader import load_document
    from ontograph.ingest.chunker import chunk
    from ontograph.utils.io import save

    _section("Step 1 — Convert + Chunk")

    raw = load_document(doc_path)
    console.print(
        f"  Format   : [cyan]{raw.source_format}[/cyan]\n"
        f"  Markdown : [cyan]{len(raw.markdown):,}[/cyan] chars"
    )
    if raw.page_map:
        console.print(f"  Pages    : [cyan]{len(raw.page_map)}[/cyan]")

    # Save raw Markdown for inspection (skip if already a .md file to avoid overwrite)
    if doc_path.suffix.lower() != ".md":
        md_out = DATA / "raw" / (doc_path.stem + ".md")
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(raw.markdown, encoding="utf-8")
        _generated("Converted Markdown", md_out)

    art = chunk(raw)
    n_chunks = len(art.chunks)
    console.print(
        f"  Total chunks : [cyan]{n_chunks}[/cyan]"
        + (f"  (will extract first {max_chunks})" if max_chunks and max_chunks < n_chunks else "")
    )

    art_path = save(art, DATA / "artifacts")
    _generated("DocumentArtifact", art_path)

    return raw, art, art_path


# ---------------------------------------------------------------------------
# Step 2 — Extract + Map
# ---------------------------------------------------------------------------

def step_extract_map(
    art,
    provider,
    tbox_path: Path | None,
    max_chunks: int | None,
) -> tuple:
    """Entity extraction + OWL triple mapping."""
    from ontograph.ingest.extractor import extract
    from ontograph.ingest.mapper import map_to_delta
    from ontograph.utils.io import save

    _section("Step 2 — Extract + Map")

    # Optionally limit extraction
    if max_chunks and len(art.chunks) > max_chunks:
        target_art = art.model_copy(update={"chunks": art.chunks[:max_chunks]})
        console.print(f"  [yellow]Limiting to first {max_chunks} chunks[/yellow]")
    else:
        target_art = art

    # TBox summary
    tbox = None
    if tbox_path is not None:
        from ontograph.utils.owl import read_tbox_summary
        tbox = read_tbox_summary(tbox_path, fmt="xml")
        console.print(
            f"  TBox     : [dim]{tbox_path.name}[/dim]  "
            f"[cyan]{len(tbox.classes)}[/cyan] classes  "
            f"[cyan]{len(tbox.datatype_properties)}[/cyan] datatype props"
        )

    # Extract
    console.print(f"\n  Extracting entities from [cyan]{len(target_art.chunks)}[/cyan] chunks…")
    bundle = extract(target_art, provider, tbox=tbox)
    console.print(f"  Entities found : [cyan]{len(bundle.entities)}[/cyan]")

    bundle_path = save(bundle, DATA / "extractions")
    _generated("ExtractionBundle", bundle_path)

    # Map
    console.print(f"\n  Mapping [cyan]{len(bundle.entities)}[/cyan] entities to OWL triples…")
    delta = map_to_delta(
        bundle,
        provider,
        namespace=tbox.namespace if tbox else "http://example.org/aerospace#",
        tbox=tbox,
    )
    console.print(f"  Proposed triples : [cyan]{len(delta.entries)}[/cyan]")

    delta_path = save(delta, DATA / "deltas")
    _generated("OntologyDelta", delta_path)

    return bundle, delta, bundle_path, delta_path


# ---------------------------------------------------------------------------
# Step 3 — Align (deduplicate entity mentions)
# ---------------------------------------------------------------------------

def step_align(
    bundle,
    delta,
    delta_path: Path,
    provider,
) -> tuple:
    """Detect and merge duplicate entity mentions."""
    from ontograph.ingest.aligner import align, apply_decisions
    from ontograph.utils.io import save

    _section("Step 3 — Align (deduplicate entities)")

    alignment = align(bundle, delta, provider=provider)
    approved_decisions = alignment.approved_decisions()

    console.print(
        f"  Candidates    : [cyan]{len(alignment.candidates)}[/cyan]\n"
        f"  Auto-approved : [green]{sum(1 for d in alignment.decisions if d.status == 'approved')}[/green]\n"
        f"  Pending       : [yellow]{sum(1 for d in alignment.decisions if d.status == 'proposed')}[/yellow]\n"
        f"  Rejected      : [red]{sum(1 for d in alignment.decisions if d.status == 'rejected')}[/red]"
    )

    if approved_decisions:
        # ── Build merge display table + report ────────────────────────────
        cand_by_id    = {c.id: c for c in alignment.candidates}
        entity_by_id  = {e.id: e for e in bundle.entities}

        mtable = Table("canonical", "merged alias(es)", "method", "score",
                       show_lines=False, box=box.SIMPLE)
        report_rows: list[str] = []

        for decision in approved_decisions:
            cand         = cand_by_id.get(decision.candidate_id)
            canon_ent    = entity_by_id.get(decision.canonical_entity_id)
            canon_surface = canon_ent.text_span if canon_ent else decision.canonical_entity_id
            merged_surfaces = [s for s in decision.aliases if s != canon_surface]
            if not merged_surfaces:
                merged_surfaces = decision.aliases
            merged_str   = ", ".join(merged_surfaces)
            method_str   = cand.method.value if cand else "?"
            score_str    = f"{cand.similarity_score:.2f}" if cand else "?"

            mtable.add_row(
                canon_surface[:45], merged_str[:45], method_str, score_str
            )
            report_rows.append(
                f"| {canon_surface} | {merged_str} | {method_str} | {score_str} |"
            )

        console.print(mtable)

        # ── Save markdown merge report ─────────────────────────────────────
        merge_dir  = DATA / "alignments"
        merge_dir.mkdir(parents=True, exist_ok=True)
        merge_path = merge_dir / f"{alignment.id}_merges.md"
        merge_path.write_text(
            f"# Entity Merge Report\n\n"
            f"- Delta  : `{delta.id}`\n"
            f"- Merges : {len(approved_decisions)}\n\n"
            f"| Canonical | Merged alias(es) | Method | Score |\n"
            f"|-----------|-----------------|--------|-------|\n"
            + "\n".join(report_rows) + "\n",
            encoding="utf-8",
        )
        _generated("Merge report", merge_path)

        aligned_delta = apply_decisions(delta, alignment)
        delta_path.write_text(aligned_delta.model_dump_json(indent=2), encoding="utf-8")
        console.print(
            f"  Applied [green]{len(approved_decisions)}[/green] merge decision(s) — "
            f"delta updated in place"
        )

        align_path = save(alignment, DATA / "alignments")
        _generated("AlignmentBundle", align_path)

        return aligned_delta, delta_path
    else:
        console.print("\n  [dim]No duplicates found — delta unchanged.[/dim]")
        return delta, delta_path


# ---------------------------------------------------------------------------
# Step 4a — Auto-approve (skips human review)
# ---------------------------------------------------------------------------

def step_auto_approve(delta, delta_path: Path) -> object:
    """Set all proposed entries to approved."""
    from ontograph.models.ontology import OntologyDelta

    _section("Step 4 — Auto-Approve")

    n_proposed = sum(1 for e in delta.entries if e.status == "proposed")
    if n_proposed == 0:
        console.print("  [dim]No proposed entries — nothing to approve.[/dim]")
        return delta

    updated_entries = []
    for entry in delta.entries:
        if entry.status == "proposed":
            updated_entries.append(entry.model_copy(update={"status": "approved"}))
        else:
            updated_entries.append(entry)

    approved_delta = delta.model_copy(update={"entries": updated_entries})
    delta_path.write_text(approved_delta.model_dump_json(indent=2), encoding="utf-8")

    console.print(
        f"  [green]✓ Auto-approved {n_proposed} proposed triple(s)[/green]\n"
        f"  Total approved : [cyan]{len(approved_delta.approved_entries())}[/cyan]"
    )
    _generated("Updated delta", delta_path)

    return approved_delta


# ---------------------------------------------------------------------------
# Step 4b — Prompt for manual review (no --auto-approve)
# ---------------------------------------------------------------------------

def step_prompt_manual_review(delta_path: Path, tbox_path: Path | None, owl_path: Path) -> None:
    """Tell the user to run review_delta.py and stop."""
    _section("Step 4 — Human Review Required")

    tbox_arg = f" \\\n        --tbox {_rel(tbox_path)}" if tbox_path else ""
    console.print(
        f"  Pipeline paused.  Review proposed triples with:\n\n"
        f"    [bold cyan].venv/Scripts/python scripts/review_delta.py[/bold cyan] \\\n"
        f"        {_rel(delta_path)}{tbox_arg} \\\n"
        f"        --owl  {_rel(owl_path)}\n\n"
        f"  Keys: [green]a[/green]=approve  [red]r[/red]=reject  "
        f"[yellow]e[/yellow]=edit  [dim]s[/dim]=skip  [dim]q[/dim]=quit\n"
        f"  Run with [white]--auto-approve[/white] to skip this step."
    )


# ---------------------------------------------------------------------------
# Step 5 — Write approved triples to OWL
# ---------------------------------------------------------------------------

def step_write_owl(
    delta,
    tbox_path: Path | None,
    owl_path: Path,
    fresh: bool = False,
) -> None:
    """Write TBox + approved ABox triples to working OWL."""
    from ontograph.utils import owl as owl_utils

    _section("Step 5 — Write to OWL")

    if fresh or not owl_path.exists():
        g = owl_utils.empty_graph()
        action = "fresh" if fresh and owl_path.exists() else "new"
        console.print(f"  Starting {action} OWL file: [dim]{owl_path.name}[/dim]")
    else:
        g = owl_utils.load_graph(owl_path, fmt="xml")
        console.print(f"  Loaded existing OWL (appending): [dim]{owl_path.name}[/dim]")

    if tbox_path is not None:
        n = owl_utils.copy_tbox(tbox_path, g, fmt="xml")
        console.print(f"  Merged TBox hierarchy: [cyan]{n}[/cyan] triples from [dim]{tbox_path.name}[/dim]")

    approved = delta.approved_entries()
    owl_utils.add_entries(g, approved)
    owl_path.parent.mkdir(parents=True, exist_ok=True)
    owl_utils.save_graph(g, owl_path, fmt="xml")

    console.print(f"  [green]✓ Wrote {len(approved)} approved ABox triple(s)[/green]")
    _generated("Working OWL", owl_path)


# ---------------------------------------------------------------------------
# Step 6 — Compare (--from-owl mode only)
# ---------------------------------------------------------------------------

def step_compare(source_owl: Path, working_owl: Path) -> None:
    """Compare the source OWL against the newly built working OWL via the Evaluator."""
    from ontograph.evaluator import evaluate

    _section("Step 6 — Comparison: source OWL vs working OWL")

    report = evaluate(source_owl, working_owl, fmt="xml")
    m_ind = report.individuals
    m_tri = report.triples

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def _color_count(val: int, good: bool = True) -> str:
        if val == 0:
            return "[dim]0[/dim]"
        color = "green" if good else "yellow"
        return f"[{color}]{val}[/{color}]"

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Metric",          style="bold")
    table.add_column("Individuals",     justify="right")
    table.add_column("Triples",         justify="right")

    table.add_row("Source count",       f"[cyan]{m_ind.source_count}[/cyan]",   f"[cyan]{m_tri.source_count}[/cyan]")
    table.add_row("Working count",      f"[cyan]{m_ind.working_count}[/cyan]",  f"[cyan]{m_tri.working_count}[/cyan]")
    table.add_row("Recovered",          _color_count(m_ind.recovered_count),    _color_count(m_tri.recovered_count))
    table.add_row("Missed",             _color_count(m_ind.missed_count, False), _color_count(m_tri.missed_count, False))
    table.add_row("Extra (invented)",   _color_count(m_ind.extra_count,  False), _color_count(m_tri.extra_count,  False))
    table.add_row("Recall",             f"[bold]{_pct(m_ind.recall)}[/bold]",   f"[bold]{_pct(m_tri.recall)}[/bold]")
    table.add_row("Precision",          f"[bold]{_pct(m_ind.precision)}[/bold]",f"[bold]{_pct(m_tri.precision)}[/bold]")
    table.add_row("F1",                 f"[bold]{_pct(m_ind.f1)}[/bold]",       f"[bold]{_pct(m_tri.f1)}[/bold]")

    console.print(table)

    if report.missed_individuals:
        console.print("[yellow]Missed individuals:[/yellow]")
        for iri in report.missed_individuals:
            local = iri.rsplit("#", 1)[-1] if "#" in iri else iri.rsplit("/", 1)[-1]
            console.print(f"  [dim]{local}[/dim]")

    if report.extra_individuals:
        console.print("[dim]Extra individuals (LLM invented):[/dim]")
        for iri in report.extra_individuals:
            local = iri.rsplit("#", 1)[-1] if "#" in iri else iri.rsplit("/", 1)[-1]
            console.print(f"  [dim]{local}[/dim]")

    # Save evaluation report to disk
    eval_dir = DATA / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"{report.id}_eval.json"
    eval_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _generated("Evaluation report", eval_path)


# ---------------------------------------------------------------------------
# Step 7 — Check rules (optional, when --rules is provided)
# ---------------------------------------------------------------------------

def step_check_rules(
    rules_path: Path,
    owl_path: Path | None,
    doc_path: Path | None,
    provider,
    mode: str,
) -> None:
    """Run LLM rule violation checker and print results."""
    from ontograph.rules import (
        load_rules,
        generate_all_plain_english,
        check_rules,
    )

    _section("Step 7 — Rule Violation Check")

    rules = load_rules(rules_path)
    console.print(
        f"  Rules file : [dim]{rules_path.name}[/dim]  "
        f"[cyan]{len(rules)}[/cyan] rule(s)  mode=[cyan]{mode}[/cyan]"
    )

    # Generate plain-English descriptions for document/both modes
    if mode in ("document", "both"):
        console.print("  Generating vague plain-English descriptions…")
        rules = generate_all_plain_english(rules, provider)

    report = check_rules(
        rules=rules,
        provider=provider,
        working_owl=owl_path,
        document_path=doc_path,
        mode=mode,
    )

    actual = report.critical()
    console.print(
        f"  Pairs evaluated : [cyan]{len(report.violations)}[/cyan]\n"
        f"  Violations found: "
        + (f"[red bold]{len(actual)}[/red bold]" if actual else "[green]0[/green]")
    )

    for vi in actual:
        sev_color = {"critical": "red", "warning": "yellow", "info": "cyan"}.get(vi.severity, "white")
        obj_part = f" ↔ {vi.object_label}" if vi.object_label else ""
        console.print(
            f"  [{sev_color}]{vi.rule_id}[/{sev_color}]  "
            f"[{sev_color}]{vi.severity}[/{sev_color}]  "
            f"[dim]({vi.mode})[/dim]  "
            f"{vi.subject_label}{obj_part}  "
            f"conf={vi.confidence:.2f}"
        )

    viol_dir = DATA / "violations"
    viol_dir.mkdir(parents=True, exist_ok=True)
    viol_path = viol_dir / f"{report.id}_violations.json"
    viol_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _generated("Violation report", viol_path)


# ---------------------------------------------------------------------------
# Step 8 — Impact analysis (optional, when --impact is provided)
# ---------------------------------------------------------------------------

def step_analyze_impact(
    scenarios_path: Path,
    rules_path: Path,
    owl_path: Path,
    doc_path: Path | None,
    provider,
    mode: str,
) -> None:
    """Run design-change impact analysis and print per-scenario P/R/F1 results."""
    from ontograph.impact import load_scenarios, analyze_impact
    from ontograph.rules import load_rules, generate_all_plain_english

    _section("Step 8 — Impact Analysis")

    namespace, scenarios = load_scenarios(scenarios_path)
    console.print(
        f"  Scenarios : [dim]{scenarios_path.name}[/dim]  "
        f"[cyan]{len(scenarios)}[/cyan] scenario(s)  mode=[cyan]{mode}[/cyan]"
    )

    rules = load_rules(rules_path)

    if mode in ("document", "both"):
        console.print("  Generating vague plain-English descriptions…")
        rules = generate_all_plain_english(rules, provider)

    eval_dir = DATA / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        console.print(f"\n  [dim]{scenario.id}[/dim] — {scenario.description.strip()[:70]}")
        result = analyze_impact(
            scenario=scenario,
            namespace=namespace,
            rules=rules,
            rules_file=str(rules_path),
            provider=provider,
            working_owl=owl_path,
            document_path=doc_path,
            mode=mode,
        )

        for arm in result.arms:
            console.print(
                f"    {arm.arm:12s}  P={arm.precision:.2f}  R={arm.recall:.2f}  F1={arm.f1:.2f}"
                + ("  ← winner" if result.winner == arm.arm else "")
            )

        out_path = eval_dir / f"{result.id}_impact.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        _generated("Impact result", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end Ontograph pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input source ──────────────────────────────────────────────────────
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input", metavar="DOC_PATH",
        help="Source document (PDF, TXT, or MD) to process",
    )
    src.add_argument(
        "--from-owl", metavar="OWL_PATH",
        help="Synthesize a document from an OWL file, then run the full pipeline",
    )

    # ── Ontology paths ────────────────────────────────────────────────────
    parser.add_argument(
        "--tbox", metavar="OWL_PATH", default=None,
        help="TBox OWL file for vocabulary-aware mapping (RDF/XML). "
             "Defaults to --from-owl value when in validation-loop mode.",
    )
    parser.add_argument(
        "--owl", metavar="OWL_PATH",
        default="data/ontology/working.owl",
        help="Output working OWL file (default: data/ontology/working.owl)",
    )

    # ── LLM + pipeline options ────────────────────────────────────────────
    parser.add_argument(
        "--provider", default="claude",
        choices=["claude", "gpt-4o", "gemini"],
        help="LLM provider (default: claude)",
    )
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="Approve all proposed triples automatically (skip human review)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None, metavar="N",
        help="Limit extraction to first N chunks (useful for quick tests)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help=(
            "Start with an empty working OWL -- ignore any existing file at --owl. "
            "Automatically enabled in --from-owl (validation-loop) mode."
        ),
    )

    # ── Rules checking (optional) ─────────────────────────────────────────
    parser.add_argument(
        "--rules", metavar="YAML", default=None,
        help="Path to a rules YAML file — runs rule violation check after OWL is written",
    )
    parser.add_argument(
        "--rules-mode", choices=["ontology", "document", "both"], default="ontology",
        help="Rule checking mode (default: ontology)",
    )

    # ── Impact analysis (optional) ────────────────────────────────────────
    parser.add_argument(
        "--impact", metavar="YAML", default=None,
        help="Path to an impact scenarios YAML file — runs impact analysis after rule check",
    )
    parser.add_argument(
        "--impact-mode", choices=["ontology", "document", "both"], default="both",
        help="Impact analysis mode (default: both)",
    )

    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────
    from_owl  = Path(args.from_owl)  if args.from_owl else None
    input_doc = Path(args.input)     if args.input    else None
    owl_path  = Path(args.owl)       if not Path(args.owl).is_absolute() else Path(args.owl)
    if not owl_path.is_absolute():
        owl_path = ROOT / owl_path

    # Default tbox to --from-owl file in validation-loop mode
    tbox_path = Path(args.tbox) if args.tbox else from_owl

    # Validation-loop always starts fresh so old data doesn't contaminate comparison
    fresh = args.fresh or (from_owl is not None)

    # ── Validate inputs ───────────────────────────────────────────────────
    if from_owl and not from_owl.exists():
        console.print(f"[red]OWL file not found: {from_owl}[/red]")
        sys.exit(1)
    if input_doc and not input_doc.exists():
        console.print(f"[red]Document not found: {input_doc}[/red]")
        sys.exit(1)
    if tbox_path and not tbox_path.exists():
        console.print(f"[red]TBox file not found: {tbox_path}[/red]")
        sys.exit(1)

    # ── API key check ─────────────────────────────────────────────────────
    _check_api_key(args.provider)

    # ── Print run summary ─────────────────────────────────────────────────
    mode = "Validation loop (OWL -> synthesize -> ingest)" if from_owl else "Document -> ontology"
    console.print(Panel(
        f"[bold]Ontograph Pipeline[/bold]\n"
        f"Mode      : [cyan]{mode}[/cyan]\n"
        f"Provider  : [cyan]{args.provider}[/cyan]\n"
        + (f"Source OWL: [dim]{from_owl}[/dim]\n" if from_owl else f"Document  : [dim]{input_doc}[/dim]\n")
        + (f"TBox      : [dim]{tbox_path}[/dim]\n" if tbox_path else "")
        + f"Output OWL: [dim]{owl_path}[/dim]\n"
        + f"Review    : [{'red]SKIPPED — auto-approve' if args.auto_approve else 'yellow]REQUIRED — will pause after alignment'}[/{'red' if args.auto_approve else 'yellow'}]\n"
        + f"Fresh OWL : {'[green]yes (old data discarded)[/green]' if fresh else '[dim]no (appending to existing)[/dim]'}",
        expand=False,
    ))

    from ontograph.llm import get_provider as _get_provider
    try:
        provider = _get_provider(args.provider)
    except Exception as exc:
        console.print(f"[red]Failed to initialise provider: {exc}[/red]")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 0 — Synthesize (validation-loop only)
    # ──────────────────────────────────────────────────────────────────────
    if from_owl:
        doc_path = step_synthesize(from_owl, provider)
    else:
        doc_path = input_doc

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1 — Convert + Chunk
    # ──────────────────────────────────────────────────────────────────────
    _raw, art, _art_path = step_ingest(doc_path, args.max_chunks)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2 — Extract + Map
    # ──────────────────────────────────────────────────────────────────────
    bundle, delta, _bundle_path, delta_path = step_extract_map(
        art, provider, tbox_path, args.max_chunks
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3 — Align
    # ──────────────────────────────────────────────────────────────────────
    delta, delta_path = step_align(bundle, delta, delta_path, provider)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4 — Review
    # ──────────────────────────────────────────────────────────────────────
    if args.auto_approve:
        delta = step_auto_approve(delta, delta_path)
    else:
        step_prompt_manual_review(delta_path, tbox_path, owl_path)
        console.print()
        sys.exit(0)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5 — Write OWL
    # ──────────────────────────────────────────────────────────────────────
    step_write_owl(delta, tbox_path, owl_path, fresh=fresh)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6 — Compare (validation-loop only)
    # ──────────────────────────────────────────────────────────────────────
    if from_owl:
        step_compare(from_owl, owl_path)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 7 — Rule violation check (optional, when --rules provided)
    # ──────────────────────────────────────────────────────────────────────
    if args.rules:
        rules_path = Path(args.rules)
        if not rules_path.is_absolute():
            rules_path = ROOT / rules_path
        if not rules_path.exists():
            console.print(f"[red]Rules file not found: {rules_path}[/red]")
            sys.exit(1)
        # Use the synthesized document for document/both mode
        check_doc = doc_path if args.rules_mode in ("document", "both") else None
        step_check_rules(rules_path, owl_path, check_doc, provider, args.rules_mode)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 8 — Impact analysis (optional, when --impact provided)
    # ──────────────────────────────────────────────────────────────────────
    if args.impact:
        impact_path = Path(args.impact)
        if not impact_path.is_absolute():
            impact_path = ROOT / impact_path
        if not impact_path.exists():
            console.print(f"[red]Impact scenarios file not found: {impact_path}[/red]")
            sys.exit(1)
        if not args.rules:
            console.print("[red]--rules is required when using --impact[/red]")
            sys.exit(1)
        impact_doc = doc_path if args.impact_mode in ("document", "both") else None
        step_analyze_impact(
            impact_path, rules_path, owl_path, impact_doc, provider, args.impact_mode
        )

    # ──────────────────────────────────────────────────────────────────────
    # Done — print summary of all generated files
    # ──────────────────────────────────────────────────────────────────────
    _section("Done")

    table = Table("Step", "File", box=box.SIMPLE, show_header=True)
    table.add_column("Step",  style="dim",  no_wrap=True)
    table.add_column("File",  style="cyan", no_wrap=False)

    if from_owl:
        syn_md = DATA / "raw" / f"{from_owl.stem}_synthesized.md"
        if syn_md.exists():
            table.add_row("0  Synthesized text",  _rel(syn_md))

    art_files  = sorted((DATA / "artifacts").glob("*.json"),  key=lambda p: p.stat().st_mtime)
    ext_files  = sorted((DATA / "extractions").glob("*.json"), key=lambda p: p.stat().st_mtime)
    dlt_files  = sorted((DATA / "deltas").glob("*.json"),      key=lambda p: p.stat().st_mtime)
    aln_files  = sorted((DATA / "alignments").glob("*.json"),  key=lambda p: p.stat().st_mtime) \
                 if (DATA / "alignments").exists() else []

    if art_files:  table.add_row("1  DocumentArtifact",  _rel(art_files[-1]))
    if ext_files:  table.add_row("2a ExtractionBundle",  _rel(ext_files[-1]))
    if dlt_files:  table.add_row("2b OntologyDelta",     _rel(dlt_files[-1]))
    if aln_files:  table.add_row("3  AlignmentBundle",   _rel(aln_files[-1]))
    if owl_path.exists():
        table.add_row("5  Working OWL",       _rel(owl_path))

    console.print(table)


if __name__ == "__main__":
    main()
