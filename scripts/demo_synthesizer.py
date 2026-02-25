"""
scripts/demo_synthesizer.py — Synthesize a design document from OWL triples.

Usage:
    # 1. Copy .env.example to .env and fill in your API key, then:

    # Synthesize from a real OWL file (ABox individuals → document):
    python scripts/demo_synthesizer.py \\
        --owl  data/ontology/cubesatontology.owl \\
        --provider claude --save

    # Use the built-in hardcoded sample delta (no OWL file needed):
    python scripts/demo_synthesizer.py                        # Claude (default)
    python scripts/demo_synthesizer.py --provider gpt-4o      # OpenAI
    python scripts/demo_synthesizer.py --save                  # save .md to scripts/output/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Make sure the project root is on the path when run directly
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env automatically so users don't have to set env vars manually
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

OUTPUT_DIR = Path(__file__).parent / "output"

def strip_citations(markdown: str) -> str:
    """Remove all [T-NNN] citation anchors and clean up extra whitespace."""
    clean = re.sub(r"\s*\[T-\d{3}\]", "", markdown)
    # Collapse any double spaces left behind
    clean = re.sub(r"  +", " ", clean)
    return clean


from ontograph.llm import get_provider
from ontograph.models.ontology import (
    ChangeSource,
    OntologyDelta,
    OntologyDeltaEntry,
    OntologyTriple,
)
from ontograph.synthesizer import attach_self_check, format_self_check_report, generate

console = Console()


# ---------------------------------------------------------------------------
# Sample delta — a small aerospace propulsion subsystem
# ---------------------------------------------------------------------------

def make_sample_delta() -> OntologyDelta:
    """Build a realistic but small OntologyDelta for a propulsion subsystem."""

    def entry(eid, subject, predicate, obj, datatype=None):
        return OntologyDeltaEntry(
            id=eid,
            triple=OntologyTriple(
                subject=subject, predicate=predicate,
                object=obj, datatype=datatype,
            ),
            rationale="Extracted from propulsion PDR document",
            confidence=0.95,
            change_source=ChangeSource.PIPELINE,
            status="approved",
        )

    AERO = "http://example.org/aerospace#"
    XSD  = "http://www.w3.org/2001/XMLSchema#"

    return OntologyDelta(
        id="demo-delta-001",
        extraction_bundle_id="demo-bundle-001",
        base_ontology_iri=AERO,
        created_at="2025-01-01T00:00:00+00:00",
        entries=[
            # ── Thruster module ───────────────────────────────────────────
            entry("t01", f"{AERO}ThrusterModule_A",
                  "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                  f"{AERO}PropulsionSubsystem"),
            entry("t02", f"{AERO}ThrusterModule_A",
                  f"{AERO}hasDryMass",         "12.4",  f"{XSD}float"),
            entry("t03", f"{AERO}ThrusterModule_A",
                  f"{AERO}hasVacuumThrust",    "220",   f"{XSD}float"),
            entry("t04", f"{AERO}ThrusterModule_A",
                  f"{AERO}hasSpecificImpulse",  "315",   f"{XSD}float"),
            entry("t05", f"{AERO}ThrusterModule_A",
                  f"{AERO}hasTRLLevel",         "6",     f"{XSD}integer"),
            entry("t06", f"{AERO}ThrusterModule_A",
                  f"{AERO}hasPropellant",
                  f"{AERO}Propellant_MON3_MMH"),
            entry("t07", f"{AERO}ThrusterModule_A",
                  f"{AERO}hasDesignLife",       "10",    f"{XSD}integer"),

            # ── Propellant tank ───────────────────────────────────────────
            entry("t08", f"{AERO}PropellantTank_A",
                  "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                  f"{AERO}StorageComponent"),
            entry("t09", f"{AERO}PropellantTank_A",
                  f"{AERO}hasTankVolume",       "85.0",  f"{XSD}float"),
            entry("t10", f"{AERO}PropellantTank_A",
                  f"{AERO}hasMaxOperatingPressure", "310", f"{XSD}float"),
            entry("t11", f"{AERO}PropellantTank_A",
                  f"{AERO}hasDryMass",          "4.2",   f"{XSD}float"),
            entry("t12", f"{AERO}PropellantTank_A",
                  f"{AERO}hasMaterial",
                  f"{AERO}Material_TitaniumAlloy"),

            # ── Feed system ───────────────────────────────────────────────
            entry("t13", f"{AERO}FeedSystem_A",
                  "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                  f"{AERO}FluidSubsystem"),
            entry("t14", f"{AERO}FeedSystem_A",
                  f"{AERO}hasNominalFlowRate",  "0.07",  f"{XSD}float"),
            entry("t15", f"{AERO}FeedSystem_A",
                  f"{AERO}hasPressureDrop",     "2.5",   f"{XSD}float"),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: synthesize a document from sample OWL triples")
    parser.add_argument("--provider", default="claude",
                        choices=["claude", "gpt-4o", "gemini"],
                        help="LLM provider to use (default: claude)")
    parser.add_argument("--model", default=None,
                        help="Override the default model for the chosen provider")
    parser.add_argument("--owl", default=None, metavar="OWL_PATH",
                        help="Path to an OWL file whose ABox individuals are synthesized "
                             "(default: use built-in sample delta)")
    parser.add_argument("--owl-format", default="xml",
                        choices=["xml", "turtle", "n3"],
                        help="rdflib format for --owl (default: xml)")
    parser.add_argument("--title", default=None,
                        help="Document title (default: derived from OWL filename or 'Propulsion Subsystem Design Description')")
    parser.add_argument("--save", action="store_true",
                        help="Save clean .md output to scripts/output/")
    args = parser.parse_args()

    # ── Provider ──────────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Provider:[/bold cyan] {args.provider}"
                  + (f"  model: {args.model}" if args.model else ""))

    # Check API key before handing off to the SDK (gives a cleaner error)
    _key_map = {
        "claude":    ("ANTHROPIC_API_KEY", "https://console.anthropic.com/"),
        "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/"),
        "gpt-4o":    ("OPENAI_API_KEY",    "https://platform.openai.com/api-keys"),
        "openai":    ("OPENAI_API_KEY",    "https://platform.openai.com/api-keys"),
        "gemini":    ("GOOGLE_API_KEY",    "https://aistudio.google.com/app/apikey"),
    }
    if args.provider in _key_map:
        env_var, url = _key_map[args.provider]
        if not os.getenv(env_var):
            console.print(
                f"\n[red bold]Missing API key: {env_var}[/red bold]\n\n"
                f"  1. Open [cyan].env[/cyan] in the project root\n"
                f"  2. Set:  {env_var}=your-key-here\n"
                f"  3. Get a key at: [cyan]{url}[/cyan]\n"
            )
            sys.exit(1)

    try:
        provider = get_provider(args.provider, model=args.model)
    except Exception as exc:
        console.print(f"[red]Failed to initialize provider: {exc}[/red]")
        sys.exit(1)

    # ── Delta ─────────────────────────────────────────────────────────────
    if args.owl:
        from ontograph.utils import owl as owl_utils

        owl_path = Path(args.owl)
        if not owl_path.exists():
            console.print(f"[red]OWL file not found: {owl_path}[/red]")
            sys.exit(1)

        console.print(f"[bold cyan]OWL:[/bold cyan] {owl_path.name}  "
                      f"(format: {args.owl_format})")
        g = owl_utils.load_graph(owl_path, fmt=args.owl_format)
        tbox = owl_utils.read_tbox_summary(owl_path, fmt=args.owl_format)
        delta = owl_utils.owl_to_delta(
            g,
            delta_id=owl_path.stem,
            base_iri=tbox.namespace,
        )
        default_title = " ".join(
            w.capitalize() for w in owl_path.stem.replace("-", " ").replace("_", " ").split()
        ) + " Design Description"
    else:
        delta = make_sample_delta()
        default_title = "Propulsion Subsystem Design Description"

    title = args.title if args.title else default_title
    approved = delta.approved_entries()
    console.print(f"[bold cyan]Delta:[/bold cyan] {len(approved)} approved entries "
                  f"across {len({e.triple.subject for e in approved})} subjects\n")

    # ── Generate ──────────────────────────────────────────────────────────
    console.print("[bold yellow]Generating document…[/bold yellow]")
    try:
        doc = generate(delta, provider, title=title)
    except Exception as exc:
        console.print(f"[red]Generation failed: {exc}[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Generated[/green] — "
                  f"{len(doc.provenance)} paragraphs, "
                  f"{len(doc.triples_cited())} triples cited\n")

    # ── Self-check ────────────────────────────────────────────────────────
    console.print("[bold yellow]Running self-check…[/bold yellow]")
    doc = attach_self_check(doc, delta)
    sc = doc.self_check
    color = "green" if sc.coverage == 1.0 else ("yellow" if sc.coverage >= 0.8 else "red")
    console.print(f"[{color}]{format_self_check_report(sc)}[/{color}]\n")

    # ── Display document (citations stripped for readability) ─────────────
    clean_markdown = strip_citations(doc.markdown)
    console.print(Panel("[bold]Generated Document[/bold]", expand=False))
    console.print(Markdown(clean_markdown))

    # ── Save clean .md to scripts/output/ ─────────────────────────────────
    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Derive a filename from the title
        slug = re.sub(r"[^\w]+", "_", title.lower()).strip("_")
        out_path = OUTPUT_DIR / f"{slug}.md"
        out_path.write_text(clean_markdown, encoding="utf-8")
        console.print(f"\n[green]Saved → {out_path}[/green]")


if __name__ == "__main__":
    main()
