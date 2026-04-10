"""
scripts/generate_ontology.py — Aerospace OWL ontology generator.

Generates a self-contained OWL ontology with a complete TBox (aerospace class
hierarchy + properties) and a populated ABox (one LLM-generated system with
realistic subsystems, components, and attributes).

No interaction with the ingest/synthesizer/evaluator pipeline.

Usage
-----
    # Predefined domain — one CubeSat design with ~15 component instances
    python scripts/generate_ontology.py --domain cubesat --provider claude

    # Scale up — larger design with ~30 component instances
    python scripts/generate_ontology.py --domain cubesat --count 30 --provider claude

    # Custom domain — LLM picks appropriate taxonomy classes
    python scripts/generate_ontology.py --domain hypersonic_vehicle --count 20 --provider claude

    # Specify output path and namespace
    python scripts/generate_ontology.py \\
        --domain rocket \\
        --count 25 \\
        --output data/ontology/my_rocket.owl \\
        --namespace http://example.org/rockets# \\
        --provider gpt-4o

Predefined domains (focused prompts): cubesat, uam, rocket, lunar
Custom domains (full vocabulary): any other string
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
DATA = ROOT / "data"
DEFAULT_NAMESPACE = "http://example.org/aerospace#"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a populated aerospace OWL ontology from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--domain", required=True, metavar="NAME",
        help=(
            "Aerospace domain to generate (e.g. cubesat, uam, rocket, lunar, "
            "hypersonic_vehicle, space_station, military_uav, …)."
        ),
    )
    parser.add_argument(
        "--count", type=int, default=15, metavar="N",
        help=(
            "Target number of component instances within the single system design "
            "(default: 15). Higher values produce more detailed ontologies."
        ),
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Output .owl file path (default: data/ontology/<domain>_generated.owl).",
    )
    parser.add_argument(
        "--provider", default="claude", choices=["claude", "gpt-4o", "gemini"],
        help="LLM provider (default: claude).",
    )
    parser.add_argument(
        "--namespace", default=DEFAULT_NAMESPACE, metavar="IRI",
        help=f"Ontology namespace IRI (default: {DEFAULT_NAMESPACE}).",
    )
    args = parser.parse_args()

    domain = args.domain.strip()
    if not domain:
        console.print("[red]--domain cannot be empty[/red]")
        sys.exit(1)
    if args.count < 1:
        console.print("[red]--count must be >= 1[/red]")
        sys.exit(1)

    output_path = (
        Path(args.output)
        if args.output
        else DATA / "ontology" / f"{domain}_generated.owl"
    )

    _key_map = {
        "claude":  "ANTHROPIC_API_KEY",
        "gpt-4o":  "OPENAI_API_KEY",
        "gemini":  "GOOGLE_API_KEY",
    }
    env_var = _key_map[args.provider]
    if not os.getenv(env_var):
        console.print(f"[red]Missing {env_var} — set it in .env[/red]")
        sys.exit(1)

    from ontograph.generator import AEROSPACE_TAXONOMY, PREDEFINED_DOMAINS
    from ontograph.generator import build_owl_graph, serialize_owl
    from ontograph.generator.instance_gen import generate_system
    from ontograph.llm import get_provider

    is_predefined = domain in PREDEFINED_DOMAINS
    domain_note = (
        "[green]predefined[/green] (focused vocabulary + subsystem guidance)"
        if is_predefined
        else "[yellow]custom[/yellow] (LLM selects classes from full taxonomy)"
    )

    console.print(Panel(
        f"[bold]Aerospace Ontology Generator[/bold]\n"
        f"Domain     : [cyan]{domain}[/cyan]  ({domain_note})\n"
        f"Components : ~[cyan]{args.count}[/cyan] total instances (target)\n"
        f"Provider   : [cyan]{args.provider}[/cyan]\n"
        f"Namespace  : [dim]{args.namespace}[/dim]\n"
        f"Output     : [dim]{output_path}[/dim]",
        expand=False,
    ))

    if not is_predefined:
        console.print(
            f"  [dim]Custom domain — the LLM will choose the most appropriate classes\n"
            f"  from the full taxonomy vocabulary for a real-world '{domain}' system.[/dim]\n"
        )

    try:
        provider = get_provider(args.provider)
    except Exception as exc:
        console.print(f"[red]Failed to initialise provider: {exc}[/red]")
        sys.exit(1)

    # ── Generate single system ────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Generating System[/bold cyan]", style="cyan"))
    console.print(f"  Generating {domain} system (~{args.count} component instances) …", end=" ")

    try:
        sys_obj = generate_system(
            domain=domain,
            taxonomy=AEROSPACE_TAXONOMY,
            provider=provider,
            namespace=args.namespace,
            instance_count=args.count,
        )
        console.print(f"[green]{sys_obj.local_name}[/green]")
    except Exception as exc:
        console.print(f"[red]FAILED: {exc}[/red]")
        sys.exit(1)

    # ── Build and serialize OWL graph ─────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Building OWL Graph[/bold cyan]", style="cyan"))
    graph = build_owl_graph([sys_obj], AEROSPACE_TAXONOMY, args.namespace)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialize_owl(graph, output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_subsystems = len(sys_obj.subsystems)
    n_components = sum(len(sub.components) for sub in sys_obj.subsystems)
    n_triples    = len(graph)

    console.print()
    console.print(Rule("[bold cyan]Summary[/bold cyan]", style="cyan"))

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Key",   style="dim",  no_wrap=True)
    table.add_column("Value", style="bold")
    table.add_row("Domain",         f"[cyan]{domain}[/cyan]")
    table.add_row("System",         f"[cyan]{sys_obj.local_name}[/cyan]")
    table.add_row("Subsystems",     str(n_subsystems))
    table.add_row("Components",     str(n_components))
    table.add_row("Total triples",  str(n_triples))
    try:
        rel = output_path.relative_to(ROOT)
    except ValueError:
        rel = output_path
    table.add_row("Saved",          f"[green]{rel}[/green]")
    console.print(table)


if __name__ == "__main__":
    main()
