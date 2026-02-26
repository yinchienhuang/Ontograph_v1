"""
scripts/analyze_impact.py — Design-change impact analysis for Ontograph.

Applies a structured design change to a working OWL and measures how accurately
each checking arm (ontology vs document) detects the resulting rule violation state.

Research question
-----------------
After modifying the ontology to reflect a design change, the ontology arm sees the
updated structured data; the document arm still reads the original (now stale) prose.
Which arm more accurately reflects the true post-change violation state?

Usage
-----
    # Both arms (recommended — shows research comparison):
    python scripts/analyze_impact.py \\
        --scenarios data/impact_scenarios_example.yaml \\
        --rules     data/rules_example.yaml \\
        --owl       data/ontology/working.owl \\
        --doc       data/raw/cubesatontology_synthesized.md \\
        --provider  claude \\
        --mode      both

    # Ontology arm only:
    python scripts/analyze_impact.py \\
        --scenarios data/impact_scenarios_example.yaml \\
        --rules     data/rules_example.yaml \\
        --owl       data/ontology/working.owl \\
        --provider  claude \\
        --mode      ontology
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


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _f1_cell(f1: float) -> str:
    if f1 >= 0.9:
        return f"[green bold]{f1:.2f}[/green bold]"
    if f1 >= 0.6:
        return f"[yellow]{f1:.2f}[/yellow]"
    return f"[red]{f1:.2f}[/red]"


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _show_results(results: list) -> None:
    """Print the impact analysis summary table and per-scenario breakdown."""

    # ── Summary table ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Impact Analysis — Summary[/bold cyan]", style="cyan"))

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Scenario",   style="dim",  no_wrap=True)
    table.add_column("Description", max_width=36)
    table.add_column("Baseline",   justify="center", no_wrap=True)
    table.add_column("GT After",   justify="center", no_wrap=True)
    table.add_column("Ont. F1",    justify="center", no_wrap=True)
    table.add_column("Doc. F1",    justify="center", no_wrap=True)
    table.add_column("Winner",     justify="center", no_wrap=True)

    ont_wins = doc_wins = ties = 0
    for r in results:
        desc    = (r.description.strip()[:34] + "…") if len(r.description.strip()) > 36 else r.description.strip()
        n_base  = len(r.baseline_violations)
        n_gt    = len(r.ground_truth_violations)
        ont_arm = r.arm("ontology")
        doc_arm = r.arm("document")

        ont_cell = _f1_cell(ont_arm.f1) if ont_arm else "[dim]—[/dim]"
        doc_cell = _f1_cell(doc_arm.f1) if doc_arm else "[dim]—[/dim]"

        if r.winner == "ontology":
            ont_wins += 1
            winner_cell = "[green bold]ontology[/green bold]"
        elif r.winner == "document":
            doc_wins += 1
            winner_cell = "[yellow bold]document[/yellow bold]"
        else:
            ties += 1
            winner_cell = "[dim]tied[/dim]"

        table.add_row(r.scenario_id, desc, str(n_base), str(n_gt), ont_cell, doc_cell, winner_cell)

    console.print(table)

    total = len(results)
    console.print(
        f"  [bold]{total}[/bold] scenario(s)  ·  "
        + (f"[green]ontology wins {ont_wins}[/green]" if ont_wins else "[dim]ontology wins 0[/dim]")
        + "  ·  "
        + (f"[yellow]document wins {doc_wins}[/yellow]" if doc_wins else "[dim]document wins 0[/dim]")
        + "  ·  "
        + (f"[dim]{ties} tied[/dim]" if ties else "[dim]0 tied[/dim]")
    )

    # ── Per-scenario detailed breakdown ───────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Detailed Breakdown[/bold cyan]", style="cyan"))

    for r in results:
        desc_short = r.description.strip()[:90]
        console.print(f"\n[bold]─── {r.scenario_id}:[/bold] {desc_short}")
        console.print(
            f"  Baseline violations : [dim]{', '.join(r.baseline_violations) or 'none'}[/dim]\n"
            f"  Ground truth after  : [cyan]{', '.join(r.ground_truth_violations) or 'none (all resolved)'}[/cyan]"
        )

        for arm in r.arms:
            detected_str = ", ".join(arm.violations_after) or "none"
            winner_tag = "  [green bold]← WINNER[/green bold]" if r.winner == arm.arm else ""
            console.print(
                f"\n  [bold]{arm.arm.upper():12s}[/bold] detected : [dim]{detected_str}[/dim]\n"
                f"               P={arm.precision:.2f}  R={arm.recall:.2f}  F1={_f1_cell(arm.f1)}"
                + winner_tag
            )

        if r.winner is None and len(r.arms) == 2:
            console.print("  [dim]Both arms tied.[/dim]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Design-change impact analysis for Ontograph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenarios", required=True, metavar="YAML",
        help="Impact scenarios YAML file",
    )
    parser.add_argument(
        "--rules", required=True, metavar="YAML",
        help="Rules YAML file (same format as check_rules.py)",
    )
    parser.add_argument(
        "--owl", required=True, metavar="OWL",
        help="Working OWL file (never modified — changes go to a temp copy)",
    )
    parser.add_argument(
        "--doc", default=None, metavar="DOC",
        help="Synthesized document text file (required for document/both mode)",
    )
    parser.add_argument(
        "--provider", default="claude",
        choices=["claude", "gpt-4o", "gemini"],
        help="LLM provider (default: claude)",
    )
    parser.add_argument(
        "--mode", default="both",
        choices=["ontology", "document", "both"],
        help="Checking mode (default: both)",
    )
    parser.add_argument(
        "--save-dir", default=None, metavar="DIR",
        help="Directory to save results JSON (default: data/evaluations)",
    )
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    rules_path     = Path(args.rules)
    owl_path       = Path(args.owl)
    doc_path       = Path(args.doc) if args.doc else None
    save_dir       = Path(args.save_dir) if args.save_dir else DATA / "evaluations"

    # ── Validate inputs ───────────────────────────────────────────────────────
    for p, label in [
        (scenarios_path, "Scenarios"),
        (rules_path,     "Rules"),
        (owl_path,       "OWL"),
    ]:
        if not p.exists():
            console.print(f"[red]{label} file not found: {p}[/red]")
            sys.exit(1)

    if args.mode in ("document", "both") and doc_path is None:
        console.print("[red]--doc is required for document/both mode[/red]")
        sys.exit(1)
    if doc_path and not doc_path.exists():
        console.print(f"[red]Document file not found: {doc_path}[/red]")
        sys.exit(1)

    _key_map = {
        "claude":  "ANTHROPIC_API_KEY",
        "gpt-4o":  "OPENAI_API_KEY",
        "gemini":  "GOOGLE_API_KEY",
    }
    env_var = _key_map.get(args.provider)
    if env_var and not os.getenv(env_var):
        console.print(f"[red]Missing {env_var} — set it in .env[/red]")
        sys.exit(1)

    # ── Print header ──────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Ontograph Impact Analysis[/bold]\n"
        f"Scenarios: [dim]{scenarios_path}[/dim]\n"
        f"Rules    : [dim]{rules_path}[/dim]\n"
        f"Mode     : [cyan]{args.mode}[/cyan]\n"
        f"Provider : [cyan]{args.provider}[/cyan]\n"
        f"OWL      : [dim]{owl_path}[/dim]\n"
        + (f"Document : [dim]{doc_path}[/dim]\n" if doc_path else ""),
        expand=False,
    ))

    # ── Load inputs ───────────────────────────────────────────────────────────
    from ontograph.impact import load_scenarios, analyze_impact
    from ontograph.rules import load_rules, generate_all_plain_english
    from ontograph.llm import get_provider

    try:
        namespace, scenarios = load_scenarios(scenarios_path)
    except Exception as exc:
        console.print(f"[red]Failed to parse scenarios YAML: {exc}[/red]")
        sys.exit(1)
    console.print(f"  Loaded [cyan]{len(scenarios)}[/cyan] scenario(s)")

    try:
        rules = load_rules(rules_path)
    except Exception as exc:
        console.print(f"[red]Failed to parse rules YAML: {exc}[/red]")
        sys.exit(1)
    console.print(f"  Loaded [cyan]{len(rules)}[/cyan] rule(s)")

    try:
        provider = get_provider(args.provider)
    except Exception as exc:
        console.print(f"[red]Failed to initialise provider: {exc}[/red]")
        sys.exit(1)

    # Generate plain-English descriptions once (needed for document/both mode)
    if args.mode in ("document", "both"):
        console.print()
        console.print(Rule(
            "[bold yellow]Generating plain-English rule descriptions[/bold yellow]",
            style="yellow",
        ))
        console.print("  [dim](deliberately vague — simulating documentation without exact specs)[/dim]")
        rules = generate_all_plain_english(rules, provider)
        for rule in rules:
            snippet = rule.plain_english[:75]
            console.print(
                f"  [dim]{rule.id}[/dim] → {snippet}"
                + ("…" if len(rule.plain_english) > 75 else "")
            )

    # ── Run analysis for each scenario ────────────────────────────────────────
    results = []
    for i, scenario in enumerate(scenarios, 1):
        console.print()
        console.print(Rule(
            f"[bold yellow]Scenario {i}/{len(scenarios)} — {scenario.id}[/bold yellow]",
            style="yellow",
        ))
        console.print(f"  [dim]{scenario.description.strip()[:100]}[/dim]")
        console.print("  Running baseline + post-change rule checks…")

        try:
            result = analyze_impact(
                scenario=scenario,
                namespace=namespace,
                rules=rules,
                rules_file=str(rules_path),
                provider=provider,
                working_owl=owl_path,
                document_path=doc_path,
                mode=args.mode,
            )
        except Exception as exc:
            console.print(f"[red]  Failed: {exc}[/red]")
            continue

        results.append(result)

        # Save immediately after each scenario
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{result.id}_impact.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        try:
            rel = out_path.relative_to(ROOT)
        except ValueError:
            rel = out_path
        console.print(f"  [dim]→ Saved: {rel}[/dim]")

    if not results:
        console.print("[red]No scenarios completed successfully.[/red]")
        sys.exit(1)

    # ── Show final results ────────────────────────────────────────────────────
    _show_results(results)


if __name__ == "__main__":
    main()
