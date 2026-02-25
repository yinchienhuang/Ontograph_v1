"""
scripts/test_ingest.py — Test the ingest pipeline on a real document.

Phase 1 (always runs): converter + chunker — no LLM, instant.
Phase 2 (opt-in):      extractor + mapper  — requires an API key.

Output files (always written):
    data/artifacts/<id>.json   — DocumentArtifact  (chunks)
    data/extractions/<id>.json — ExtractionBundle  (Phase 2 only)
    data/deltas/<id>.json      — OntologyDelta     (Phase 2 only)
    data/raw/<stem>.md         — converted Markdown (always, for inspection)

Usage:
    # Phase 1 only (no API calls):
    python scripts/test_ingest.py data/raw/cds_rev13_final2.pdf

    # Phase 1 + 2 (LLM extraction + mapping):
    python scripts/test_ingest.py data/raw/cds_rev13_final2.pdf --extract

    # TBox-aware mapping (uses cubesatontology.owl vocabulary):
    python scripts/test_ingest.py data/raw/cds_rev13_final2.pdf --extract \\
        --tbox data/ontology/cubesatontology.owl

    # Limit extraction to first N chunks (cheaper quick test):
    python scripts/test_ingest.py data/raw/cds_rev13_final2.pdf --extract --max-chunks 5

    # Choose LLM provider:
    python scripts/test_ingest.py data/raw/cds_rev13_final2.pdf --extract --provider gpt-4o
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

console = Console()

DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Phase 1 — Converter + Chunker
# ---------------------------------------------------------------------------

def run_phase1(path: Path, max_chunks: int | None = None):
    from ontograph.ingest.loader import load_document
    from ontograph.ingest.chunker import chunk
    from ontograph.utils.io import save

    console.print(Panel(
        f"[bold]Phase 1 — Convert + Chunk[/bold]\n[dim]{path}[/dim]",
        expand=False,
    ))

    # ── Convert ───────────────────────────────────────────────────────────
    console.print("\n[bold yellow]Converting…[/bold yellow]")
    raw = load_document(path)
    console.print(f"  Format   : [cyan]{raw.source_format}[/cyan]")
    console.print(f"  SHA-256  : [dim]{raw.source_sha256[:16]}…[/dim]")
    console.print(f"  Markdown : [cyan]{len(raw.markdown):,}[/cyan] chars")
    if raw.page_map:
        console.print(f"  Pages    : [cyan]{len(raw.page_map)}[/cyan]")

    # Save converted markdown alongside the source for easy inspection
    md_out = DATA / "raw" / (Path(path).stem + ".md")
    md_out.write_text(raw.markdown, encoding="utf-8")
    console.print(f"  [dim]Markdown saved → {md_out.relative_to(ROOT)}[/dim]")

    # ── Chunk ─────────────────────────────────────────────────────────────
    console.print("\n[bold yellow]Chunking…[/bold yellow]")
    art = chunk(raw)
    display = art.chunks[:max_chunks] if max_chunks else art.chunks
    console.print(
        f"  Total chunks : [cyan]{len(art.chunks)}[/cyan]"
        + (f"  (showing first {max_chunks})" if max_chunks else "")
    )

    # Save DocumentArtifact
    art_path = save(art, DATA / "artifacts")
    console.print(f"  [dim]Artifact  saved → {art_path.relative_to(ROOT)}[/dim]")

    # ── Display chunk table ───────────────────────────────────────────────
    table = Table("idx", "page/line", "tokens", "section", "text preview",
                  show_lines=False)
    for i, c in enumerate(display):
        loc = c.source_locator
        pos = f"p{loc.page}" if loc.page else f"L{loc.line_start}-{loc.line_end}"
        preview = " ".join(c.text.split()[:14])
        if len(c.text.split()) > 14:
            preview += "…"
        section = c.section_context
        if len(section) > 45:
            section = section[:42] + "…"
        table.add_row(str(i), pos, str(c.token_count), section, preview)
    console.print(table)

    return raw, art


# ---------------------------------------------------------------------------
# Phase 2 — Extractor + Mapper
# ---------------------------------------------------------------------------

def run_phase2(
    art,
    provider_name: str,
    max_chunks: int | None = None,
    tbox_path: Path | None = None,
):
    from ontograph.llm import get_provider
    from ontograph.ingest.extractor import extract
    from ontograph.ingest.mapper import map_to_delta
    from ontograph.utils.io import save

    console.print(Panel("[bold]Phase 2 — Extract + Map[/bold]", expand=False))

    # ── API key check ─────────────────────────────────────────────────────
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

    provider = get_provider(provider_name)
    console.print(
        f"  Provider : [cyan]{provider_name}[/cyan]  "
        f"model=[cyan]{provider.model_id}[/cyan]"
    )

    # ── Optionally limit to first N chunks ────────────────────────────────
    if max_chunks and len(art.chunks) > max_chunks:
        target_art = art.model_copy(update={"chunks": art.chunks[:max_chunks]})
        console.print(f"  [yellow]Limiting extraction to first {max_chunks} chunks[/yellow]")
    else:
        target_art = art

    # ── Extract ───────────────────────────────────────────────────────────
    console.print(
        f"\n[bold yellow]Extracting entities "
        f"({len(target_art.chunks)} chunks)…[/bold yellow]"
    )
    bundle = extract(target_art, provider)
    console.print(f"  Entities found : [cyan]{len(bundle.entities)}[/cyan]")

    bundle_path = save(bundle, DATA / "extractions")
    console.print(f"  [dim]Bundle saved → {bundle_path.relative_to(ROOT)}[/dim]")

    if bundle.entities:
        etable = Table("entity", "type", "conf", "attributes", "section",
                       show_lines=False)
        for e in bundle.entities:
            attrs = ", ".join(
                f"{a.name}={a.value}{a.unit or ''}" for a in e.attributes[:3]
            )
            if len(e.attributes) > 3:
                attrs += f" (+{len(e.attributes)-3} more)"
            section = (e.section_context[:38] + "…"
                       if len(e.section_context) > 38 else e.section_context)
            etable.add_row(
                e.text_span[:35], e.entity_type,
                f"{e.confidence:.2f}", attrs or "(none)", section,
            )
        console.print(etable)

    # ── Load TBox (optional) ──────────────────────────────────────────────
    tbox = None
    if tbox_path is not None:
        from ontograph.utils.owl import read_tbox_summary
        console.print(f"\n[bold yellow]Loading TBox…[/bold yellow]")
        tbox = read_tbox_summary(tbox_path, fmt="xml")
        console.print(
            f"  Classes            : [cyan]{len(tbox.classes)}[/cyan]"
        )
        console.print(
            f"  Object properties  : [cyan]{len(tbox.object_properties)}[/cyan]"
        )
        console.print(
            f"  Datatype properties: [cyan]{len(tbox.datatype_properties)}[/cyan]"
        )
        console.print(f"  Namespace: [dim]{tbox.namespace}[/dim]")

    # ── Map ───────────────────────────────────────────────────────────────
    console.print("\n[bold yellow]Mapping to OWL triples…[/bold yellow]")
    delta = map_to_delta(
        bundle,
        provider,
        namespace=tbox.namespace if tbox else "http://example.org/aerospace#",
        tbox=tbox,
    )
    console.print(f"  Proposed triples : [cyan]{len(delta.entries)}[/cyan]")

    delta_path = save(delta, DATA / "deltas")
    console.print(f"  [dim]Delta    saved → {delta_path.relative_to(ROOT)}[/dim]")

    if delta.entries:
        ttable = Table("subject", "predicate", "object", "datatype", "conf",
                       show_lines=False)
        for e in delta.entries:
            t = e.triple
            s  = t.subject.rsplit("#", 1)[-1][:30]
            p  = t.predicate.rsplit("#", 1)[-1][:25]
            o  = t.object.rsplit("#", 1)[-1][:25]
            dt = (t.datatype or "").rsplit("#", 1)[-1]
            ttable.add_row(s, p, o, dt, f"{e.confidence:.2f}")
        console.print(ttable)

    return bundle, delta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the ingest pipeline on a document file."
    )
    parser.add_argument("path", help="Path to the source document (PDF, TXT, or MD)")
    parser.add_argument(
        "--extract", action="store_true",
        help="Run Phase 2: LLM entity extraction + OWL mapping",
    )
    parser.add_argument(
        "--provider", default="gpt-4o",
        choices=["claude", "gpt-4o", "gemini"],
        help="LLM provider for Phase 2 (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="Limit chunk display / extraction to first N chunks",
    )
    parser.add_argument(
        "--tbox",
        metavar="OWL_PATH",
        default=None,
        help=(
            "Path to a TBox OWL file (RDF/XML). When provided the mapper is "
            "instructed to prefer its class and property vocabulary. "
            "Also sets the mapping namespace to the TBox namespace."
        ),
    )
    args = parser.parse_args()

    doc_path = Path(args.path)
    if not doc_path.exists():
        console.print(f"[red]File not found: {doc_path}[/red]")
        sys.exit(1)

    _raw, art = run_phase1(doc_path, max_chunks=args.max_chunks)

    tbox_path = Path(args.tbox) if args.tbox else None
    if tbox_path and not tbox_path.exists():
        console.print(f"[red]TBox file not found: {tbox_path}[/red]")
        sys.exit(1)

    if args.extract:
        run_phase2(art, args.provider, max_chunks=args.max_chunks, tbox_path=tbox_path)
    else:
        console.print(
            "\n[dim]Run with [white]--extract[/white] "
            "to run LLM extraction + mapping.[/dim]"
        )


if __name__ == "__main__":
    main()
