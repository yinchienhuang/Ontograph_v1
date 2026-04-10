"""
scripts/detect_conflicts.py — Autonomous conflict detection (no pre-written rules).

Compares two arms:
  ontograph  — OWL -> LLM discovers rules -> LLM checks exact values  (two-pass)
  direct     — raw document -> LLM finds conflicts in one shot         (one-pass)

Usage
-----
    # Both arms (full comparison):
    python scripts/detect_conflicts.py \\
        --owl  data/ontology/cubesatontology.owl \\
        --doc  data/raw/cubesatontology_synthesized.md \\
        --provider gpt-4o

    # Ontograph arm only:
    python scripts/detect_conflicts.py \\
        --owl  data/ontology/uam.owl \\
        --provider claude \\
        --mode ontograph

    # Direct arm only:
    python scripts/detect_conflicts.py \\
        --doc  data/raw/rocket_design_synthesized.md \\
        --provider gpt-4o \\
        --mode direct
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console(highlight=False)
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _severity_color(s: str) -> str:
    return {"critical": "red", "warning": "yellow", "info": "cyan"}.get(s, "white")


def _show_conflicts(conflicts, arm_label: str, color: str) -> None:
    if not conflicts:
        console.print(f"\n  [dim]{arm_label}: no conflicts detected[/dim]")
        return

    console.print(f"\n[bold {color}]{arm_label}[/bold {color}]  "
                  f"({len(conflicts)} conflict{'s' if len(conflicts) != 1 else ''} found)")

    for i, c in enumerate(conflicts, 1):
        pair = f"{c.subject} <-> {c.object}" if c.object else c.subject
        sev_c = _severity_color(c.severity)
        console.print(
            f"\n  [{sev_c}][{i}] {c.conflict_type}[/{sev_c}]  "
            f"[bold]{pair}[/bold]  "
            f"[dim]conf={c.confidence:.2f}  sev={c.severity}[/dim]"
        )
        console.print(f"      {c.description}")
        for ev in c.evidence:
            console.print(f"      [dim]Evidence: {ev[:120]}[/dim]")


def _show_comparison(report) -> None:
    """Side-by-side summary table + per-arm details."""
    ont = report.ontograph_conflicts
    doc = report.direct_conflicts

    # ── Summary table ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Conflict Detection Comparison[/bold cyan]", style="cyan"))

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Arm",           style="dim",   no_wrap=True)
    table.add_column("Conflicts",     justify="right")
    table.add_column("Critical",      justify="right")
    table.add_column("Warning",       justify="right")
    table.add_column("Input tok.",    justify="right")
    table.add_column("Output tok.",   justify="right")
    table.add_column("Total tok.",    justify="right")

    def _row(label, conflicts, tokens):
        crit = sum(1 for c in conflicts if c.severity == "critical")
        warn = sum(1 for c in conflicts if c.severity == "warning")
        return (
            label,
            str(len(conflicts)),
            f"[red]{crit}[/red]" if crit else "[dim]0[/dim]",
            f"[yellow]{warn}[/yellow]" if warn else "[dim]0[/dim]",
            str(tokens["input"]),
            str(tokens["output"]),
            f"[cyan]{tokens['input'] + tokens['output']}[/cyan]",
        )

    if ont is not None:
        table.add_row(*_row("Ontograph (2-pass)", ont, report.ontograph_tokens))
    if doc is not None:
        table.add_row(*_row("Direct LLM (1-pass)", doc, report.direct_tokens))

    console.print(table)

    # ── Per-arm details ────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Detailed Conflicts[/bold cyan]", style="cyan"))

    if report.mode in ("ontograph", "both"):
        _show_conflicts(ont, "ONTOGRAPH ARM", "green")

    if report.mode in ("direct", "both"):
        _show_conflicts(doc, "DIRECT LLM ARM", "blue")

    # ── Overlap analysis (both mode only) ─────────────────────────────────────
    if report.mode == "both" and ont and doc:
        console.print()
        console.print(Rule("[bold cyan]Overlap Analysis[/bold cyan]", style="cyan"))

        ont_subjects = {(c.subject, c.object or "") for c in ont}
        doc_subjects = {(c.subject, c.object or "") for c in doc}

        ont_only = ont_subjects - doc_subjects
        doc_only = doc_subjects - ont_subjects
        shared   = ont_subjects & doc_subjects

        console.print(f"\n  Both found         : [green]{len(shared)}[/green]")
        console.print(f"  Ontograph only     : [yellow]{len(ont_only)}[/yellow]")
        console.print(f"  Direct LLM only    : [yellow]{len(doc_only)}[/yellow]")

        if ont_only:
            console.print("\n  [yellow]Ontograph-only conflicts (missed by direct LLM):[/yellow]")
            for subj, obj in sorted(ont_only):
                console.print(f"    {subj}" + (f" <-> {obj}" if obj else ""))
        if doc_only:
            console.print("\n  [yellow]Direct-LLM-only conflicts (not found by ontograph):[/yellow]")
            for subj, obj in sorted(doc_only):
                console.print(f"    {subj}" + (f" <-> {obj}" if obj else ""))


# ---------------------------------------------------------------------------
# Scoring display
# ---------------------------------------------------------------------------

def _outcome_cell(outcome: str) -> str:
    return {
        "TP": "[green]TP[/green]",
        "FP": "[red]FP[/red]",
        "FN": "[yellow]FN[/yellow]",
        "TN": "[dim]TN[/dim]",
    }.get(outcome, outcome)


def _show_score_report(sr, mode: str) -> None:
    """Display P/R/F1 summary + per-rule breakdown tables."""
    console.print()
    console.print(Rule("[bold magenta]Ground-Truth Evaluation[/bold magenta]", style="magenta"))

    # ── Summary metrics table ──────────────────────────────────────────────────
    summary = Table(box=box.SIMPLE, show_header=True, title="[bold]Arm Metrics[/bold]")
    summary.add_column("Arm",        style="dim",  no_wrap=True)
    summary.add_column("Precision",  justify="right")
    summary.add_column("Recall",     justify="right")
    summary.add_column("F1",         justify="right")
    summary.add_column("TP",         justify="right")
    summary.add_column("FP",         justify="right")
    summary.add_column("FN",         justify="right")
    summary.add_column("TN",         justify="right")
    summary.add_column("Tokens in",  justify="right")
    summary.add_column("Tokens out", justify="right")

    def _summary_row(label, precision, recall, f1, per_rule, tokens):
        tp = sum(1 for s in per_rule if s.outcome == "TP")
        fp = sum(1 for s in per_rule if s.outcome == "FP")
        fn = sum(1 for s in per_rule if s.outcome == "FN")
        tn = sum(1 for s in per_rule if s.outcome == "TN")
        f1_color = "green" if f1 >= 0.7 else ("yellow" if f1 >= 0.4 else "red")
        return (
            label,
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"[{f1_color}]{f1:.2f}[/{f1_color}]",
            f"[green]{tp}[/green]" if tp else "[dim]0[/dim]",
            f"[red]{fp}[/red]"     if fp else "[dim]0[/dim]",
            f"[yellow]{fn}[/yellow]" if fn else "[dim]0[/dim]",
            f"[dim]{tn}[/dim]",
            str(tokens.get("input", 0)),
            str(tokens.get("output", 0)),
        )

    if mode in ("ontograph", "both"):
        summary.add_row(*_summary_row(
            "Ontograph (2-pass)",
            sr.ontograph_precision, sr.ontograph_recall, sr.ontograph_f1,
            sr.per_rule_ontograph, sr.ontograph_tokens,
        ))
    if mode in ("direct", "both"):
        summary.add_row(*_summary_row(
            "Direct LLM (1-pass)",
            sr.direct_precision, sr.direct_recall, sr.direct_f1,
            sr.per_rule_direct, sr.direct_tokens,
        ))
    console.print(summary)

    # ── Per-rule breakdown ─────────────────────────────────────────────────────
    breakdown = Table(box=box.SIMPLE, show_header=True, title="[bold]Per-Rule Breakdown[/bold]")
    breakdown.add_column("Rule ID",   no_wrap=True)
    breakdown.add_column("Sev.",      no_wrap=True)
    breakdown.add_column("GT",        no_wrap=True)
    if mode in ("ontograph", "both"):
        breakdown.add_column("Ont.",  justify="center")
        breakdown.add_column("Score", justify="right")
    if mode in ("direct", "both"):
        breakdown.add_column("Dir.",  justify="center")
        breakdown.add_column("Score", justify="right")
    breakdown.add_column("Name")

    ont_map = {s.rule_id: s for s in sr.per_rule_ontograph}
    dir_map = {s.rule_id: s for s in sr.per_rule_direct}

    # Use whichever arm has data to get rule list
    rule_scores = sr.per_rule_ontograph or sr.per_rule_direct
    for rs in rule_scores:
        gt_cell  = "[red]VIOL[/red]" if rs.gt_violated else "[dim]ok[/dim]"
        sev_cell = f"[{_severity_color(rs.severity)}]{rs.severity[:4]}[/{_severity_color(rs.severity)}]"
        row: list[str] = [rs.rule_id, sev_cell, gt_cell]

        if mode in ("ontograph", "both"):
            ont_rs = ont_map.get(rs.rule_id)
            row.append(_outcome_cell(ont_rs.outcome) if ont_rs else "[dim]-[/dim]")
            row.append(f"{ont_rs.match_score:.2f}" if ont_rs and ont_rs.match_score else "[dim]-[/dim]")
        if mode in ("direct", "both"):
            dir_rs = dir_map.get(rs.rule_id)
            row.append(_outcome_cell(dir_rs.outcome) if dir_rs else "[dim]-[/dim]")
            row.append(f"{dir_rs.match_score:.2f}" if dir_rs and dir_rs.match_score else "[dim]-[/dim]")

        row.append(rs.rule_name)
        breakdown.add_row(*row)

    console.print(breakdown)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous conflict detection — no pre-written rules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--owl",      metavar="OWL_PATH", default=None,
                        help="Working OWL file (required for ontograph/both mode)")
    parser.add_argument("--doc",      metavar="DOC_PATH", default=None,
                        help="Synthesized document (required for direct/both mode)")
    parser.add_argument("--provider", default="gpt-4o",
                        choices=["claude", "gpt-4o", "gemini"],
                        help="LLM provider (default: gpt-4o)")
    parser.add_argument("--mode",     default="both",
                        choices=["ontograph", "direct", "both"],
                        help="Which arm(s) to run (default: both)")
    parser.add_argument("--rules",     metavar="YAML", default=None,
                        help="Rules YAML for deterministic ground-truth scoring "
                             "(uses expected_violated field — no extra LLM call)")
    parser.add_argument("--hint-mode", default="vague",
                        choices=["none", "vague", "exact"],
                        help="Rule knowledge given to both arms: "
                             "none=no hints, vague=plain_english, exact=full rule spec "
                             "(default: vague; only active when --rules is provided)")
    args = parser.parse_args()

    owl_path   = Path(args.owl)   if args.owl   else None
    doc_path   = Path(args.doc)   if args.doc   else None
    rules_path = Path(args.rules) if args.rules else None

    scoring_enabled = rules_path is not None

    # ── Validate ──────────────────────────────────────────────────────────────
    if rules_path and not rules_path.exists():
        console.print(f"[red]Rules file not found: {rules_path}[/red]")
        sys.exit(1)

    if args.mode in ("ontograph", "both") and owl_path is None:
        console.print("[red]--owl is required for mode='ontograph'/'both'[/red]")
        sys.exit(1)
    if args.mode in ("direct", "both") and doc_path is None:
        console.print("[red]--doc is required for mode='direct'/'both'[/red]")
        sys.exit(1)
    if owl_path and not owl_path.exists():
        console.print(f"[red]OWL file not found: {owl_path}[/red]")
        sys.exit(1)
    if doc_path and not doc_path.exists():
        console.print(f"[red]Document not found: {doc_path}[/red]")
        sys.exit(1)

    _key_map = {"claude": "ANTHROPIC_API_KEY", "gpt-4o": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY"}
    env_var = _key_map.get(args.provider)
    if env_var and not os.getenv(env_var):
        console.print(f"[red]Missing {env_var} — set it in .env[/red]")
        sys.exit(1)

    # ── Load provider ─────────────────────────────────────────────────────────
    from ontograph.llm import get_provider
    try:
        provider = get_provider(args.provider)
    except Exception as exc:
        console.print(f"[red]Failed to initialise provider: {exc}[/red]")
        sys.exit(1)

    # ── Generate hints for both arms (per hint-mode) ──────────────────────────
    ontograph_hints: list[str] | None = None
    direct_hints:    list[str] | None = None

    if rules_path and args.hint_mode != "none":
        from ontograph.rules.loader import load_rules as _load_rules
        _rules_loaded = _load_rules(rules_path)

        if args.hint_mode == "vague":
            from ontograph.rules.generator import generate_all_plain_english
            console.print()
            console.print(Rule("[bold yellow]Generating vague rule hints...[/bold yellow]", style="yellow"))
            _rules_loaded = generate_all_plain_english(_rules_loaded, provider)
            hints = [r.plain_english for r in _rules_loaded if r.plain_english]
        else:  # exact
            from ontograph.rules.conflict_detector import _format_exact_hint
            hints = [_format_exact_hint(r) for r in _rules_loaded]

        ontograph_hints = hints
        direct_hints    = hints

    # ── Header ────────────────────────────────────────────────────────────────
    hint_mode_label = args.hint_mode if rules_path else "none"
    console.print(Panel(
        f"[bold]Ontograph Conflict Detector[/bold]\n"
        f"Mode     : [cyan]{args.mode}[/cyan]\n"
        f"Provider : [cyan]{args.provider}[/cyan]\n"
        f"Hint mode: [cyan]{hint_mode_label}[/cyan]\n"
        + (f"OWL      : [dim]{owl_path}[/dim]\n" if owl_path else "")
        + (f"Document : [dim]{doc_path}[/dim]\n" if doc_path else "")
        + (f"Hints    : [dim]{len(ontograph_hints)} guidelines (both arms)[/dim]\n" if ontograph_hints else ""),
        expand=False,
    ))

    # ── Run ───────────────────────────────────────────────────────────────────
    from ontograph.rules.conflict_detector import detect_conflicts

    if args.mode in ("ontograph", "both"):
        console.print()
        console.print(Rule("[bold yellow]Ontograph arm: extracting OWL individuals[/bold yellow]", style="yellow"))

    try:
        report = detect_conflicts(
            provider=provider,
            owl_path=owl_path,
            document_path=doc_path,
            mode=args.mode,
            ontograph_hints=ontograph_hints,
            direct_hints=direct_hints,
        )
    except Exception as exc:
        console.print(f"[red]Conflict detection failed: {exc}[/red]")
        raise

    # ── Display ───────────────────────────────────────────────────────────────
    _show_comparison(report)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = DATA / "conflicts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.id}_conflicts.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    try:
        rel = out_path.relative_to(ROOT)
    except ValueError:
        rel = out_path
    console.print(f"\n  [dim]-> Conflict report:[/dim] [cyan]{rel}[/cyan]")

    # ── Ground-truth scoring (optional) ───────────────────────────────────────
    if scoring_enabled:
        from ontograph.rules.loader import load_rules as _load_rules_score
        from ontograph.rules.conflict_scorer import score_conflicts

        # GT is deterministic — read directly from expected_violated in YAML.
        # No check_rules() LLM call needed.
        rules = _load_rules_score(rules_path)
        score_report = score_conflicts(report, rules)

        _show_score_report(score_report, report.mode)

        score_path = out_dir / f"{report.id}_score.json"
        score_path.write_text(score_report.model_dump_json(indent=2), encoding="utf-8")
        try:
            rel_score = score_path.relative_to(ROOT)
        except ValueError:
            rel_score = score_path
        console.print(f"\n  [dim]-> Score report:[/dim]    [cyan]{rel_score}[/cyan]")


if __name__ == "__main__":
    main()
