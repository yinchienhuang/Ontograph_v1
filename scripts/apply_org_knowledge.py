"""
scripts/apply_org_knowledge.py — Apply a YAML org-knowledge file to a working OWL graph.

Usage:
    python scripts/apply_org_knowledge.py \\
        --yaml data/org_knowledge.yaml \\
        --owl  data/ontology/working.ttl

    # Preview without writing:
    python scripts/apply_org_knowledge.py \\
        --yaml data/org_knowledge_example.yaml \\
        --owl  data/ontology/working.ttl \\
        --dry-run
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
from rich.table import Table

from ontograph.ingest.org_loader import load_org_knowledge
from ontograph.utils.owl import add_entries, empty_graph, load_graph, save_graph

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a YAML organizational-knowledge file to a working OWL graph."
    )
    parser.add_argument(
        "--yaml", required=True,
        help="Path to the org_knowledge.yaml file",
    )
    parser.add_argument(
        "--owl", required=True,
        help="Path to the working .ttl OWL file (created if it does not exist)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the rules that would be applied without modifying the OWL file",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    owl_path  = Path(args.owl)

    if not yaml_path.exists():
        console.print(f"[red]YAML file not found: {yaml_path}[/red]")
        sys.exit(1)

    # ── Load rules ────────────────────────────────────────────────────────────
    try:
        entries = load_org_knowledge(yaml_path)
    except Exception as exc:
        console.print(f"[red]Failed to parse YAML: {exc}[/red]")
        sys.exit(1)

    console.print(
        f"\n[bold cyan]Loaded {len(entries)} organizational knowledge rules "
        f"from [white]{yaml_path.name}[/white][/bold cyan]\n"
    )

    # ── Display rules table ───────────────────────────────────────────────────
    def _short(iri: str) -> str:
        """Return the local name of an IRI (after # or last /)."""
        return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

    table = Table("ID", "Subject", "Predicate", "Object", "Note", show_lines=False)
    for e in entries:
        t = e.triple
        note = e.rationale.replace("\n", " ").strip()
        if len(note) > 65:
            note = note[:62] + "…"
        table.add_row(e.id, _short(t.subject), _short(t.predicate), _short(t.object), note)
    console.print(table)

    if args.dry_run:
        console.print("\n[yellow]Dry-run mode — OWL file not modified.[/yellow]")
        return

    # ── Load or create OWL graph ──────────────────────────────────────────────
    if owl_path.exists():
        graph = load_graph(owl_path)
        console.print(f"\n[dim]Loaded existing OWL graph ({len(graph)} triples): {owl_path}[/dim]")
    else:
        graph = empty_graph()
        console.print(f"\n[dim]No OWL file found — creating new graph at {owl_path}[/dim]")

    before = len(graph)
    add_entries(graph, entries)
    after = len(graph)
    save_graph(graph, owl_path)

    console.print(
        f"\n[green]✓ Written {after - before} new triples → {owl_path}[/green]\n"
        f"  (Graph: {before} → {after} total triples)"
    )


if __name__ == "__main__":
    main()
