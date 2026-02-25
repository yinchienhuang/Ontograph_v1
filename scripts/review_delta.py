"""
scripts/review_delta.py — Interactive human reviewer for an OntologyDelta.

For each proposed triple the reviewer sees:

    [1/12] PROPOSED
    Subject  : http://example.org/cubesat-ontology#ThrusterModule_A
    Predicate: http://www.w3.org/1999/02/22-rdf-syntax-ns#type
    Object   : http://example.org/cubesat-ontology#PropulsionSubsystem
    Datatype : —
    Rationale: Entity type inferred from extracted entity 'ThrusterModule_A'
    Confidence: 0.95

    [a]pprove  [r]eject  [e]dit object  [s]kip  [q]uit  > _

Commands
--------
a — approve the triple as-is
r — reject (status → "rejected")
e — edit the object value inline, then approve
s — leave status unchanged (stays "proposed"), skip for now
q — quit; unsaved progress is lost (changes already written are kept)

After reviewing, the updated delta JSON is saved back to the same file and
the approved triples are written to the OWL file (if --owl is given).

When --tbox is provided the output OWL is seeded with the full TBox before
approved ABox triples are added.  This preserves the original class hierarchy
(rdfs:subClassOf chains, property definitions, labels, comments) so every
class is present in the output even when no instance of it has been created.

Usage
-----
    # Review a delta file (no OWL write):
    python scripts/review_delta.py data/deltas/<id>.json

    # Review and write approved triples to OWL (ABox only):
    python scripts/review_delta.py data/deltas/<id>.json --owl data/ontology/working.owl

    # Review + write, preserving TBox hierarchy in the output OWL:
    python scripts/review_delta.py data/deltas/<id>.json \\
        --owl  data/ontology/working.owl \\
        --tbox data/ontology/cubesatontology.owl

    # Review only a specific status (default: proposed):
    python scripts/review_delta.py data/deltas/<id>.json --status proposed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import box

console = Console()

DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short(iri: str, max_len: int = 60) -> str:
    """Return a short human-readable form: local name or truncated IRI."""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1][:max_len]
    if "/" in iri:
        return iri.rsplit("/", 1)[-1][:max_len]
    return iri[:max_len]


def _display_entry(idx: int, total: int, entry) -> None:
    """Pretty-print one delta entry."""
    t = entry.triple
    status_color = {
        "proposed": "yellow",
        "approved": "green",
        "rejected": "red",
    }.get(entry.status, "white")

    console.print(
        f"\n[bold][{idx}/{total}][/bold] "
        f"[{status_color}]{entry.status.upper()}[/{status_color}]"
        f"  [dim]id={entry.id}[/dim]"
    )

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("field", style="bold cyan", width=12)
    table.add_column("value")
    table.add_row("Subject",    _short(t.subject))
    table.add_row("Predicate",  _short(t.predicate))
    table.add_row("Object",     _short(t.object))
    table.add_row("Datatype",   _short(t.datatype) if t.datatype else "—")
    table.add_row("Rationale",  entry.rationale or "—")
    table.add_row("Confidence", f"{entry.confidence:.2f}")
    console.print(table)


# ---------------------------------------------------------------------------
# Review loop
# ---------------------------------------------------------------------------

def review_loop(delta, filter_status: str):
    """
    Iterate over entries matching *filter_status* and prompt the user.

    Returns the modified delta (entries updated in place).
    """
    candidates = [e for e in delta.entries if e.status == filter_status]
    total = len(candidates)

    if total == 0:
        console.print(
            f"[yellow]No entries with status='{filter_status}' found.[/yellow]"
        )
        return delta

    console.print(
        f"\n[bold]Reviewing [cyan]{total}[/cyan] "
        f"'{filter_status}' entries[/bold]\n"
        "Commands: [green]a[/green]pprove  "
        "[red]r[/red]eject  "
        "[yellow]e[/yellow]dit object  "
        "[dim]s[/dim]kip  "
        "[bold]q[/bold]uit\n"
    )

    # Build index: entry id → position in delta.entries
    id_to_idx = {e.id: i for i, e in enumerate(delta.entries)}

    approved_count = 0
    rejected_count = 0

    for seq, entry in enumerate(candidates, start=1):
        _display_entry(seq, total, entry)

        while True:
            raw = Prompt.ask(
                "[bold]> [/bold]",
                choices=["a", "r", "e", "s", "q"],
                show_choices=False,
                console=console,
            ).strip().lower()

            if raw == "q":
                console.print(
                    f"\n[dim]Quit after reviewing {seq - 1}/{total} entries.[/dim]"
                )
                console.print(
                    f"[green]Approved:[/green] {approved_count}  "
                    f"[red]Rejected:[/red] {rejected_count}"
                )
                return delta

            if raw == "s":
                break  # leave status unchanged

            if raw == "a":
                delta.entries[id_to_idx[entry.id]] = entry.model_copy(
                    update={"status": "approved"}
                )
                approved_count += 1
                console.print("[green]✓ Approved[/green]")
                break

            if raw == "r":
                delta.entries[id_to_idx[entry.id]] = entry.model_copy(
                    update={"status": "rejected"}
                )
                rejected_count += 1
                console.print("[red]✗ Rejected[/red]")
                break

            if raw == "e":
                new_obj = Prompt.ask(
                    f"  New object value [dim](current: {_short(entry.triple.object)})[/dim]",
                    console=console,
                ).strip()
                if new_obj:
                    new_triple = entry.triple.model_copy(update={"object": new_obj})
                    delta.entries[id_to_idx[entry.id]] = entry.model_copy(
                        update={"triple": new_triple, "status": "approved"}
                    )
                    approved_count += 1
                    console.print(
                        f"[green]✓ Approved[/green] with edited object: "
                        f"[cyan]{new_obj}[/cyan]"
                    )
                else:
                    console.print("[dim]No change — skipping.[/dim]")
                break

    console.print(
        f"\n[bold]Review complete.[/bold]  "
        f"[green]Approved:[/green] {approved_count}  "
        f"[red]Rejected:[/red] {rejected_count}"
    )
    return delta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively review and approve/reject proposed OWL triples."
    )
    parser.add_argument(
        "delta_path",
        help="Path to an OntologyDelta JSON file (e.g. data/deltas/<id>.json)",
    )
    parser.add_argument(
        "--owl",
        metavar="OWL_PATH",
        default=None,
        help=(
            "OWL file to write approved triples into. "
            "If omitted, triples are only saved to the delta JSON."
        ),
    )
    parser.add_argument(
        "--owl-format",
        default="xml",
        choices=["xml", "turtle", "n3"],
        help="rdflib format for the OWL file (default: xml)",
    )
    parser.add_argument(
        "--tbox",
        metavar="TBOX_PATH",
        default=None,
        help=(
            "Path to the source TBox OWL file (RDF/XML). When provided its full "
            "class hierarchy and property definitions are merged into --owl before "
            "approved ABox triples are written, so every class remains present "
            "even if it has no instances."
        ),
    )
    parser.add_argument(
        "--tbox-format",
        default="xml",
        choices=["xml", "turtle", "n3"],
        help="rdflib format for the TBox file (default: xml)",
    )
    parser.add_argument(
        "--status",
        default="proposed",
        choices=["proposed", "approved", "rejected"],
        help="Review entries with this status (default: proposed)",
    )
    args = parser.parse_args()

    delta_path = Path(args.delta_path)
    if not delta_path.exists():
        console.print(f"[red]File not found: {delta_path}[/red]")
        sys.exit(1)

    # ── Load delta ─────────────────────────────────────────────────────────
    from ontograph.utils.io import load
    from ontograph.models.ontology import OntologyDelta

    delta = load(delta_path, OntologyDelta)
    console.print(Panel(
        f"[bold]Delta Review[/bold]\n"
        f"[dim]{delta_path}[/dim]\n"
        f"Total entries : [cyan]{len(delta.entries)}[/cyan]  |  "
        f"Proposed : [yellow]{sum(1 for e in delta.entries if e.status == 'proposed')}[/yellow]  |  "
        f"Approved : [green]{sum(1 for e in delta.entries if e.status == 'approved')}[/green]  |  "
        f"Rejected : [red]{sum(1 for e in delta.entries if e.status == 'rejected')}[/red]",
        expand=False,
    ))

    # ── Interactive review ─────────────────────────────────────────────────
    delta = review_loop(delta, filter_status=args.status)

    # ── Save updated delta ─────────────────────────────────────────────────
    delta_path.write_text(delta.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"\n[dim]Delta saved → {delta_path}[/dim]")

    # ── Write approved triples to OWL (if requested) ───────────────────────
    if args.owl:
        from ontograph.utils import owl as owl_utils

        owl_path = Path(args.owl)

        # Validate TBox path early
        tbox_path = Path(args.tbox) if args.tbox else None
        if tbox_path and not tbox_path.exists():
            console.print(f"[red]TBox file not found: {tbox_path}[/red]")
            sys.exit(1)

        if owl_path.exists():
            g = owl_utils.load_graph(owl_path, fmt=args.owl_format)
            console.print(f"\n  Loaded existing OWL: [dim]{owl_path}[/dim]")
        else:
            g = owl_utils.empty_graph()
            console.print(f"\n  Creating new OWL file: [dim]{owl_path}[/dim]")

        # Seed with TBox hierarchy (idempotent — safe to re-run)
        if tbox_path is not None:
            owl_utils.copy_tbox(tbox_path, g, fmt=args.tbox_format)
            console.print(
                f"  Merged TBox hierarchy from: [dim]{tbox_path.name}[/dim]"
            )

        approved = delta.approved_entries()
        owl_utils.add_entries(g, approved)
        owl_utils.save_graph(g, owl_path, fmt=args.owl_format)
        console.print(
            f"  [green]Wrote {len(approved)} approved ABox triple(s) → {owl_path}[/green]"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    approved_total = sum(1 for e in delta.entries if e.status == "approved")
    rejected_total = sum(1 for e in delta.entries if e.status == "rejected")
    proposed_total = sum(1 for e in delta.entries if e.status == "proposed")
    console.print(
        f"\n[bold]Final delta state:[/bold]  "
        f"[green]approved={approved_total}[/green]  "
        f"[red]rejected={rejected_total}[/red]  "
        f"[yellow]proposed={proposed_total}[/yellow]"
    )


if __name__ == "__main__":
    main()
