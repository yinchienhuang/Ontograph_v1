"""
scripts/align_delta.py — Entity deduplication for a produced OntologyDelta.

Detects that two extracted entity mentions refer to the same real-world
concept (e.g. "FAA" and "Federal Aviation Administration") and rewrites the
delta so they share a single canonical IRI, inserting skos:altLabel triples
for every alias surface form.

Runs three detection methods in priority order:
  1. ACRONYM           — one string is an initialism of the other (auto-approved)
  2. STRING_SIMILARITY — token Jaccard overlap (auto-approved if >= 0.85)
  3. LLM               — judgment call for ambiguous mid-range pairs

Usage
-----
    # Detect duplicates, show candidates, do NOT rewrite delta (dry run):
    python scripts/align_delta.py \\
        data/extractions/<bundle-id>.json \\
        data/deltas/<delta-id>.json

    # Detect + apply all approved decisions → writes aligned delta in-place:
    python scripts/align_delta.py \\
        data/extractions/<bundle-id>.json \\
        data/deltas/<delta-id>.json \\
        --apply

    # Use LLM for ambiguous pairs:
    python scripts/align_delta.py \\
        data/extractions/<bundle-id>.json \\
        data/deltas/<delta-id>.json \\
        --apply --provider claude

    # Save alignment bundle JSON alongside the delta:
    python scripts/align_delta.py \\
        data/extractions/<bundle-id>.json \\
        data/deltas/<delta-id>.json \\
        --apply --save-alignment
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

DATA = ROOT / "data"


def _short(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rsplit("/", 1)[-1]
    return iri


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and merge duplicate entity mentions in an OntologyDelta."
    )
    parser.add_argument(
        "bundle_path",
        help="Path to an ExtractionBundle JSON (data/extractions/<id>.json)",
    )
    parser.add_argument(
        "delta_path",
        help="Path to an OntologyDelta JSON (data/deltas/<id>.json)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Rewrite the delta in-place with approved alignment decisions",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["claude", "gpt-4o", "gemini"],
        help="LLM provider for ambiguous pairs (default: no LLM, leave proposed)",
    )
    parser.add_argument(
        "--save-alignment", action="store_true",
        help="Save the OntologyAlignmentBundle JSON to data/alignments/",
    )
    args = parser.parse_args()

    bundle_path = Path(args.bundle_path)
    delta_path  = Path(args.delta_path)

    for p in (bundle_path, delta_path):
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)

    # ── Load artifacts ────────────────────────────────────────────────────
    from ontograph.utils.io import load, save
    from ontograph.models.extraction import ExtractionBundle
    from ontograph.models.ontology import OntologyDelta

    bundle = load(bundle_path, ExtractionBundle)
    delta  = load(delta_path, OntologyDelta)

    console.print(Panel(
        f"[bold]Align Delta[/bold]\n"
        f"Bundle  : [dim]{bundle_path.name}[/dim]  "
        f"([cyan]{len(bundle.entities)}[/cyan] entities)\n"
        f"Delta   : [dim]{delta_path.name}[/dim]  "
        f"([cyan]{len(delta.entries)}[/cyan] entries)",
        expand=False,
    ))

    # ── Load LLM provider (optional) ─────────────────────────────────────
    provider = None
    if args.provider:
        _key_map = {
            "claude": "ANTHROPIC_API_KEY",
            "gpt-4o": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }
        env_var = _key_map[args.provider]
        if not os.getenv(env_var):
            console.print(
                f"\n[red bold]Missing {env_var}[/red bold] — "
                f"set it in [cyan].env[/cyan]\n"
            )
            sys.exit(1)
        from ontograph.llm import get_provider
        provider = get_provider(args.provider)
        console.print(
            f"  LLM provider : [cyan]{args.provider}[/cyan]  "
            f"model=[cyan]{provider.model_id}[/cyan]"
        )

    # ── Run aligner ───────────────────────────────────────────────────────
    from ontograph.ingest.aligner import align, apply_decisions

    console.print("\n[bold yellow]Detecting duplicate entity mentions…[/bold yellow]")
    alignment = align(bundle, delta, provider=provider)

    console.print(
        f"  Candidates found  : [cyan]{len(alignment.candidates)}[/cyan]\n"
        f"  Auto-approved     : [green]{sum(1 for d in alignment.decisions if d.status == 'approved')}[/green]\n"
        f"  Pending (LLM/low) : [yellow]{sum(1 for d in alignment.decisions if d.status == 'proposed')}[/yellow]\n"
        f"  Rejected by LLM   : [red]{sum(1 for d in alignment.decisions if d.status == 'rejected')}[/red]"
    )

    # ── Display candidates table ──────────────────────────────────────────
    if alignment.candidates:
        table = Table(
            "surface A", "surface B", "score", "method", "decision",
            show_lines=False, box=box.SIMPLE,
        )
        for cand in alignment.candidates:
            decision = alignment.decision_for(cand.id)
            status = decision.status if decision else "—"
            status_color = {"approved": "green", "rejected": "red", "proposed": "yellow"}.get(status, "dim")
            table.add_row(
                cand.surface_a[:30],
                cand.surface_b[:30],
                f"{cand.similarity_score:.2f}",
                cand.method.value,
                f"[{status_color}]{status}[/{status_color}]",
            )
        console.print(table)
    else:
        console.print("  [dim]No duplicate candidates found.[/dim]")

    # ── Save alignment bundle ─────────────────────────────────────────────
    if args.save_alignment:
        align_path = save(alignment, DATA / "alignments")
        console.print(f"\n[dim]Alignment saved → {align_path.relative_to(ROOT)}[/dim]")

    # ── Apply decisions to delta ──────────────────────────────────────────
    if args.apply:
        approved = alignment.approved_decisions()
        if not approved:
            console.print("\n[yellow]No approved decisions — delta unchanged.[/yellow]")
        else:
            aligned_delta = apply_decisions(delta, alignment)

            # Count rewrites and new altLabel entries
            old_subjects = {e.triple.subject for e in delta.entries}
            new_subjects  = {e.triple.subject for e in aligned_delta.entries}
            rewritten = len(old_subjects - new_subjects)
            alt_entries = sum(
                1 for e in aligned_delta.entries
                if e.triple.predicate == "http://www.w3.org/2004/02/skos/core#altLabel"
            )

            # Write back in-place
            delta_path.write_text(aligned_delta.model_dump_json(indent=2), encoding="utf-8")
            console.print(
                f"\n[green]✓ Applied {len(approved)} decision(s)[/green]\n"
                f"  IRIs rewritten  : [cyan]{rewritten}[/cyan]\n"
                f"  altLabel triples: [cyan]{alt_entries}[/cyan]\n"
                f"  [dim]Delta saved → {delta_path}[/dim]"
            )
    else:
        console.print(
            "\n[dim]Run with [white]--apply[/white] to rewrite the delta "
            "with approved decisions.[/dim]"
        )


if __name__ == "__main__":
    main()
