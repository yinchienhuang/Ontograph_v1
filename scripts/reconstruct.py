"""
scripts/reconstruct.py — Triple reconstruction comparison CLI.

Compares the onto-graph structured pipeline against a single-shot general-AI
approach for reconstructing OWL triples from a document.

Both arms are scored against the source OWL (ground truth) using triple-level
Precision / Recall / F1.

Usage
-----
    # Compare both arms with a pre-synthesized document (fastest)
    .venv/Scripts/python scripts/reconstruct.py \\
        --owl  data/ontology/cubesatontology.owl \\
        --doc  data/raw/cubesatontology_synthesized.md \\
        --provider claude \\
        --mode both

    # Auto-synthesize the document from the OWL (calls LLM synthesizer)
    .venv/Scripts/python scripts/reconstruct.py \\
        --owl  data/ontology/cubesatontology.owl \\
        --provider claude

    # Direct arm only (quick test)
    .venv/Scripts/python scripts/reconstruct.py \\
        --owl  data/ontology/cubesatontology.owl \\
        --doc  data/raw/cubesatontology_synthesized.md \\
        --provider claude \\
        --mode direct
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

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()
DATA    = ROOT / "data"


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
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triple reconstruction: onto-graph pipeline vs. direct AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--owl", required=True, metavar="PATH",
        help="Source OWL file (ground truth, RDF/XML format)",
    )
    parser.add_argument(
        "--doc", default=None, metavar="PATH",
        help="Pre-synthesized document (Markdown/TXT).  "
             "If omitted, the synthesizer generates one from the OWL.",
    )
    parser.add_argument(
        "--namespace", default=None, metavar="IRI",
        help="Ontology namespace IRI.  Auto-detected from --owl if not provided.",
    )
    parser.add_argument(
        "--provider", default="claude",
        choices=["claude", "gpt-4o", "gemini"],
        help="LLM provider (default: claude)",
    )
    parser.add_argument(
        "--mode", default="both",
        choices=["ontograph", "direct", "both"],
        help="Which arm(s) to run (default: both)",
    )
    parser.add_argument(
        "--save-dir", default=None, metavar="DIR",
        help="Directory to save the ReconstructionReport (default: data/evaluations)",
    )

    args = parser.parse_args()

    # Resolve paths
    owl_path = Path(args.owl)
    if not owl_path.is_absolute():
        owl_path = ROOT / owl_path

    doc_path = Path(args.doc) if args.doc else None
    if doc_path and not doc_path.is_absolute():
        doc_path = ROOT / doc_path

    save_dir = Path(args.save_dir) if args.save_dir else DATA / "evaluations"

    # Validate
    if not owl_path.exists():
        console.print(f"[red]OWL file not found: {owl_path}[/red]")
        sys.exit(1)
    if doc_path and not doc_path.exists():
        console.print(f"[red]Document not found: {doc_path}[/red]")
        sys.exit(1)

    _check_api_key(args.provider)

    # Print run header
    doc_label = _rel(doc_path) if doc_path else "[dim](will be synthesized from OWL)[/dim]"
    console.print(Panel(
        f"[bold]Triple Reconstruction[/bold]\n"
        f"Source OWL : [dim]{_rel(owl_path)}[/dim]\n"
        f"Document   : {doc_label}\n"
        f"Provider   : [cyan]{args.provider}[/cyan]\n"
        f"Mode       : [cyan]{args.mode}[/cyan]",
        expand=False,
    ))

    # Get provider
    from ontograph.llm import get_provider as _get_provider
    try:
        provider = _get_provider(args.provider)
    except Exception as exc:
        console.print(f"[red]Failed to initialise provider: {exc}[/red]")
        sys.exit(1)

    # Run reconstruction
    console.print()
    console.print(Rule("[bold yellow]Running Reconstruction[/bold yellow]", style="yellow"))

    from ontograph.reconstruction import run_reconstruction
    try:
        report = run_reconstruction(
            source_owl=owl_path,
            document_path=doc_path,
            provider=provider,
            namespace=args.namespace,
            mode=args.mode,
            save_dir=save_dir,
        )
    except Exception as exc:
        console.print(f"[red]Reconstruction failed: {exc}[/red]")
        raise

    # Print results table
    console.print()
    console.print(Rule("[bold yellow]Results[/bold yellow]", style="yellow"))

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Arm",        style="bold",    no_wrap=True)
    table.add_column("Indiv P",    justify="right")
    table.add_column("Indiv R",    justify="right")
    table.add_column("Indiv F1",   justify="right")
    table.add_column("Triple P",   justify="right")
    table.add_column("Triple R",   justify="right")
    table.add_column("Triple F1",  justify="right")
    table.add_column("Src triples",  justify="right")
    table.add_column("Pred triples", justify="right")

    for arm in report.arms:
        winner_marker = "  ← winner" if report.winner == arm.arm else ""
        table.add_row(
            arm.arm + winner_marker,
            _pct(arm.individual_precision),
            _pct(arm.individual_recall),
            _pct(arm.individual_f1),
            _pct(arm.triple_precision),
            _pct(arm.triple_recall),
            f"[bold]{_pct(arm.triple_f1)}[/bold]",
            str(arm.triple_count_source),
            str(arm.triple_count_predicted),
        )

    console.print(table)

    # Winner summary
    if report.winner:
        other_arms = [a for a in report.arms if a.arm != report.winner]
        winner_arm = next(a for a in report.arms if a.arm == report.winner)
        if other_arms:
            other = other_arms[0]
            console.print(
                f"[bold green]Winner: {report.winner}[/bold green]  "
                f"(triple F1: {_pct(winner_arm.triple_f1)} vs {_pct(other.triple_f1)})"
            )
        else:
            console.print(f"[bold]Arm: {report.winner}  triple F1: {_pct(winner_arm.triple_f1)}[/bold]")
    elif len(report.arms) == 2:
        console.print("[dim]Tied — no winner[/dim]")

    # Save path
    console.print()
    save_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"  Report saved → [cyan]{_rel(save_dir)}/{report.id}.json[/cyan]")


if __name__ == "__main__":
    main()
