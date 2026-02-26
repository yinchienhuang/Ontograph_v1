"""
scripts/check_rules.py — Standalone LLM rule checker for Ontograph.

Loads structured compatibility rules from a YAML file and detects violations
using LLMs in two modes:

  ontology mode  — precise structured data from the working OWL graph
  document mode  — vague auto-generated plain English + synthesized document text
  both           — runs both modes and shows a side-by-side comparison with source citations

Usage
-----
    # Ontology mode:
    python scripts/check_rules.py \\
        --rules    data/rules_example.yaml \\
        --owl      data/ontology/working.owl \\
        --provider claude \\
        --mode     ontology

    # Document mode (requires synthesized document):
    python scripts/check_rules.py \\
        --rules    data/rules_example.yaml \\
        --doc      data/raw/cubesatontology_synthesized.md \\
        --provider claude \\
        --mode     document

    # Both modes — full side-by-side comparison with source citations:
    python scripts/check_rules.py \\
        --rules    data/rules_example.yaml \\
        --owl      data/ontology/working.owl \\
        --doc      data/raw/cubesatontology_synthesized.md \\
        --provider claude \\
        --mode     both
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
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

def _short(label: str | None, n: int = 45) -> str:
    if not label:
        return "—"
    return label[:n]


def _severity_color(severity: str) -> str:
    return {"critical": "red", "warning": "yellow", "info": "cyan"}.get(severity, "white")


def _violated_cell(violated: bool, severity: str) -> str:
    if violated:
        c = _severity_color(severity)
        return f"[{c} bold]YES[/{c} bold]"
    return "[green]no[/green]"


# ---------------------------------------------------------------------------
# Single-mode flat table (ontology-only or document-only)
# ---------------------------------------------------------------------------

def _show_flat_table(report) -> None:
    """Print the standard flat violations table with a Source column."""
    all_violations = report.violations
    actual_violations = report.critical()

    console.print(
        f"\n  Total evaluated pairs : [cyan]{len(all_violations)}[/cyan]\n"
        f"  Violations detected   : "
        + (f"[red bold]{len(actual_violations)}[/red bold]" if actual_violations else "[green]0[/green]")
    )

    if not all_violations:
        console.print("\n  [dim]No pairs evaluated — check that OWL contains individuals of the required types.[/dim]")
        return

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Rule",        style="dim",  no_wrap=True)
    table.add_column("Mode",        style="dim",  no_wrap=True)
    table.add_column("Severity",    no_wrap=True)
    table.add_column("Subject",     no_wrap=True)
    table.add_column("Object",      no_wrap=True)
    table.add_column("Violated",    no_wrap=True)
    table.add_column("Conf.",       justify="right", no_wrap=True)
    table.add_column("Explanation")
    table.add_column("Source")

    for vi in all_violations:
        sev_color   = _severity_color(vi.severity)
        explanation = vi.explanation[:55] + "…" if len(vi.explanation) > 57 else vi.explanation
        source_str  = vi.source_refs[0][:55] + "…" if vi.source_refs and len(vi.source_refs[0]) > 57 else (vi.source_refs[0] if vi.source_refs else "")

        table.add_row(
            vi.rule_id,
            vi.mode,
            f"[{sev_color}]{vi.severity}[/{sev_color}]",
            _short(vi.subject_label),
            _short(vi.object_label),
            _violated_cell(vi.violated, vi.severity),
            f"{vi.confidence:.2f}",
            explanation,
            f"[dim]{source_str}[/dim]",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Side-by-side comparison view (--mode both)
# ---------------------------------------------------------------------------

def _show_comparison(report, rules) -> None:
    """
    Full side-by-side comparison for --mode both.

    Shows:
      1. Summary table — one row per rule, violation counts, agreement
      2. Stats line
      3. Per-rule detailed breakdown with source citations
    """
    from ontograph.rules.schema import ViolationInstance

    rules_by_id = {r.id: r for r in rules}

    # Group violations by rule_id then split by mode
    ont_by_rule: dict[str, list[ViolationInstance]] = defaultdict(list)
    doc_by_rule: dict[str, list[ViolationInstance]] = defaultdict(list)
    all_rule_ids: list[str] = []

    for vi in report.violations:
        if vi.rule_id not in all_rule_ids:
            all_rule_ids.append(vi.rule_id)
        if vi.mode == "ontology":
            ont_by_rule[vi.rule_id].append(vi)
        else:
            doc_by_rule[vi.rule_id].append(vi)

    # Also include rules that produced no violations at all
    for r in rules:
        if r.id not in all_rule_ids:
            all_rule_ids.append(r.id)

    # Agreement categorisation
    agree_count  = ont_only_count = doc_only_count = neither_count = 0

    def _agreement(rule_id: str) -> str:
        ont_viol = any(v.violated for v in ont_by_rule.get(rule_id, []))
        doc_viol = any(v.violated for v in doc_by_rule.get(rule_id, []))
        if ont_viol and doc_viol:
            return "agree-both"
        if not ont_viol and not doc_viol:
            return "agree-neither"
        if ont_viol:
            return "ont-only"
        return "doc-only"

    for rule_id in all_rule_ids:
        cat = _agreement(rule_id)
        if cat in ("agree-both", "agree-neither"):
            agree_count += 1
        elif cat == "ont-only":
            ont_only_count += 1
        else:
            doc_only_count += 1

    # ── Part 1: Summary table ─────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Comparison Summary[/bold cyan]", style="cyan"))

    summary = Table(box=box.SIMPLE, show_header=True)
    summary.add_column("Rule",         style="dim",  no_wrap=True)
    summary.add_column("Name",         no_wrap=False, max_width=30)
    summary.add_column("Sev.",         no_wrap=True)
    summary.add_column("Ontology",     justify="center", no_wrap=True)
    summary.add_column("Document",     justify="center", no_wrap=True)
    summary.add_column("Agreement",    no_wrap=True)

    def _count_cell(violations: list) -> str:
        n = sum(1 for v in violations if v.violated)
        total = len(violations)
        if n == 0:
            return f"[dim]0 / {total}[/dim]"
        return f"[red]{n}[/red] / {total}"

    def _agree_cell(cat: str) -> str:
        return {
            "agree-both":    "[green]✓ both violated[/green]",
            "agree-neither": "[green]✓ both clear[/green]",
            "ont-only":      "[yellow]✗ ontology-only[/yellow]",
            "doc-only":      "[yellow]✗ document-only[/yellow]",
        }[cat]

    for rule_id in all_rule_ids:
        rule     = rules_by_id.get(rule_id)
        name     = (rule.name[:28] + "…") if rule and len(rule.name) > 30 else (rule.name if rule else rule_id)
        severity = rule.severity if rule else "warning"
        sev_c    = _severity_color(severity)
        cat      = _agreement(rule_id)

        summary.add_row(
            rule_id,
            name,
            f"[{sev_c}]{severity}[/{sev_c}]",
            _count_cell(ont_by_rule.get(rule_id, [])),
            _count_cell(doc_by_rule.get(rule_id, [])),
            _agree_cell(cat),
        )

    console.print(summary)

    # ── Part 2: Stats line ────────────────────────────────────────────────────
    n_rules = len(all_rule_ids)
    console.print(
        f"  [bold]{n_rules}[/bold] rules checked  ·  "
        f"[green]{agree_count} agree[/green]  ·  "
        + (f"[yellow]{ont_only_count} ontology-only[/yellow]  ·  " if ont_only_count else "[dim]0 ontology-only[/dim]  ·  ")
        + (f"[yellow]{doc_only_count} document-only[/yellow]" if doc_only_count else "[dim]0 document-only[/dim]")
    )

    # ── Part 3: Per-rule detailed breakdown ───────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Detailed Breakdown[/bold cyan]", style="cyan"))

    for rule_id in all_rule_ids:
        rule     = rules_by_id.get(rule_id)
        name     = rule.name if rule else rule_id
        severity = rule.severity if rule else "warning"
        sev_c    = _severity_color(severity)
        cat      = _agreement(rule_id)

        console.print(
            f"\n[bold]─── {rule_id}:[/bold] {name}  "
            f"[[{sev_c}]{severity}[/{sev_c}]]"
        )

        ont_results = ont_by_rule.get(rule_id, [])
        doc_results = doc_by_rule.get(rule_id, [])

        # ── Ontology side ──────────────────────────────────────────────────
        n_ont_viol = sum(1 for v in ont_results if v.violated)
        if ont_results:
            console.print(
                f"\n  [bold]ONTOLOGY[/bold]  "
                + (f"[red]({n_ont_viol} violation{'s' if n_ont_viol != 1 else ''} found)[/red]"
                   if n_ont_viol else "[green](no violations)[/green]")
            )
            for vi in ont_results:
                pair = f"{vi.subject_label} ↔ {vi.object_label}" if vi.object_label else vi.subject_label
                status = _violated_cell(vi.violated, vi.severity)
                console.print(f"    {pair}   {status}   conf={vi.confidence:.2f}")
                console.print(f"    [dim]Reasoning:[/dim] {vi.explanation}")
                if vi.source_refs:
                    console.print(f"    [dim]Source   :[/dim] [cyan]{vi.source_refs[0]}[/cyan]")
                    for ref in vi.source_refs[1:]:
                        console.print(f"               [cyan]{ref}[/cyan]")
        else:
            console.print("\n  [bold]ONTOLOGY[/bold]  [dim](not checked / no individuals of required type found)[/dim]")

        # ── Document side ──────────────────────────────────────────────────
        n_doc_viol = sum(1 for v in doc_results if v.violated)
        if doc_results:
            console.print(
                f"\n  [bold]DOCUMENT[/bold]  "
                + (f"[red]({n_doc_viol} violation{'s' if n_doc_viol != 1 else ''} found)[/red]"
                   if n_doc_viol else "[green](no violations)[/green]")
            )
            for vi in doc_results:
                pair = f"{vi.subject_label} ↔ {vi.object_label}" if vi.object_label else vi.subject_label
                status = _violated_cell(vi.violated, vi.severity)
                console.print(f"    {pair}   {status}   conf={vi.confidence:.2f}")
                console.print(f"    [dim]Reasoning:[/dim] {vi.explanation}")
                if vi.source_refs:
                    for ref in vi.source_refs:
                        # Wrap long quotes for readability
                        wrapped = (ref[:110] + "…") if len(ref) > 113 else ref
                        console.print(f"    [dim]Source   :[/dim] [italic dim]\"{wrapped}\"[/italic dim]")
        else:
            console.print("\n  [bold]DOCUMENT[/bold]  [dim](LLM found no relevant content)[/dim]")

        # ── Agreement verdict ──────────────────────────────────────────────
        verdict = {
            "agree-both":    "[green]✓ Both methods found violations[/green]",
            "agree-neither": "[green]✓ Both methods: no violation detected[/green]",
            "ont-only":      "[yellow]✗ Disagreement: ontology detected a violation — document did not[/yellow]",
            "doc-only":      "[yellow]✗ Disagreement: document detected a violation — ontology did not[/yellow]",
        }[cat]
        console.print(f"\n  {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-based rule violation checker for Ontograph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rules", required=True, metavar="YAML",
        help="Path to rules YAML file",
    )
    parser.add_argument(
        "--owl", metavar="OWL_PATH", default=None,
        help="Working OWL file (required for ontology/both mode)",
    )
    parser.add_argument(
        "--doc", metavar="DOC_PATH", default=None,
        help="Synthesized document text file (required for document/both mode)",
    )
    parser.add_argument(
        "--provider", default="claude",
        choices=["claude", "gpt-4o", "gemini"],
        help="LLM provider (default: claude)",
    )
    parser.add_argument(
        "--mode", default="ontology",
        choices=["ontology", "document", "both"],
        help="Checking mode (default: ontology)",
    )
    args = parser.parse_args()

    rules_path = Path(args.rules)
    owl_path   = Path(args.owl)   if args.owl else None
    doc_path   = Path(args.doc)   if args.doc else None

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not rules_path.exists():
        console.print(f"[red]Rules file not found: {rules_path}[/red]")
        sys.exit(1)
    if args.mode in ("ontology", "both") and owl_path is None:
        console.print(f"[red]--owl is required for mode='{args.mode}'[/red]")
        sys.exit(1)
    if args.mode in ("document", "both") and doc_path is None:
        console.print(f"[red]--doc is required for mode='{args.mode}'[/red]")
        sys.exit(1)
    if owl_path and not owl_path.exists():
        console.print(f"[red]OWL file not found: {owl_path}[/red]")
        sys.exit(1)
    if doc_path and not doc_path.exists():
        console.print(f"[red]Document file not found: {doc_path}[/red]")
        sys.exit(1)

    # ── API key check ─────────────────────────────────────────────────────────
    _key_map = {
        "claude": "ANTHROPIC_API_KEY",
        "gpt-4o": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }
    env_var = _key_map.get(args.provider)
    if env_var and not os.getenv(env_var):
        console.print(f"[red]Missing {env_var} — set it in .env[/red]")
        sys.exit(1)

    # ── Print run summary ─────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Ontograph Rule Checker[/bold]\n"
        f"Rules   : [dim]{rules_path}[/dim]\n"
        f"Mode    : [cyan]{args.mode}[/cyan]\n"
        f"Provider: [cyan]{args.provider}[/cyan]\n"
        + (f"OWL     : [dim]{owl_path}[/dim]\n" if owl_path else "")
        + (f"Document: [dim]{doc_path}[/dim]\n" if doc_path else ""),
        expand=False,
    ))

    # ── Load rules ────────────────────────────────────────────────────────────
    from ontograph.rules import load_rules, generate_all_plain_english, check_rules

    try:
        rules = load_rules(rules_path)
    except Exception as exc:
        console.print(f"[red]Failed to parse rules YAML: {exc}[/red]")
        sys.exit(1)

    console.print(f"\n  Loaded [cyan]{len(rules)}[/cyan] rule(s) from [dim]{rules_path.name}[/dim]")

    # ── Initialise LLM provider ───────────────────────────────────────────────
    from ontograph.llm import get_provider
    try:
        provider = get_provider(args.provider)
    except Exception as exc:
        console.print(f"[red]Failed to initialise provider: {exc}[/red]")
        sys.exit(1)

    # ── Generate plain English (document/both mode) ───────────────────────────
    if args.mode in ("document", "both"):
        console.print()
        console.print(Rule("[bold yellow]Generating plain-English rule descriptions[/bold yellow]", style="yellow"))
        console.print("  [dim](deliberately vague — simulating documentation without exact specs)[/dim]")
        rules = generate_all_plain_english(rules, provider)
        for rule in rules:
            snippet = rule.plain_english[:80]
            console.print(f"  [dim]{rule.id}[/dim] → {snippet}{'…' if len(rule.plain_english) > 80 else ''}")

    # ── Run checks ────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold yellow]Checking rules[/bold yellow]", style="yellow"))

    try:
        report = check_rules(
            rules=rules,
            provider=provider,
            working_owl=owl_path,
            document_path=doc_path,
            mode=args.mode,
            rules_file=str(rules_path),
        )
    except Exception as exc:
        console.print(f"[red]Rule check failed: {exc}[/red]")
        sys.exit(1)

    # ── Display results ───────────────────────────────────────────────────────
    if args.mode == "both":
        _show_comparison(report, rules)
    else:
        _show_flat_table(report)

    # ── Save report ───────────────────────────────────────────────────────────
    viol_dir = DATA / "violations"
    viol_dir.mkdir(parents=True, exist_ok=True)
    out_path = viol_dir / f"{report.id}_violations.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    try:
        rel = out_path.relative_to(ROOT)
    except ValueError:
        rel = out_path
    console.print(f"\n  [dim]-> Violation report:[/dim] [cyan]{rel}[/cyan]")


if __name__ == "__main__":
    main()
