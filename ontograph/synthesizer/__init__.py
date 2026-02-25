"""
ontograph.synthesizer — Grounded document generation from OWL triples.

Quick start:
    from ontograph.synthesizer import generate, attach_self_check
    from ontograph.llm import get_provider

    provider = get_provider("claude")
    doc = generate(delta, provider, title="Propulsion Subsystem CONOPS")
    doc = attach_self_check(doc, delta)
    print(doc.markdown)
    print(f"Coverage: {doc.self_check.coverage:.0%}")
"""

from ontograph.synthesizer.generator import (
    SectionDraft,
    ParagraphDraft,
    build_anchor_map,
    generate,
    label_from_iri,
)
from ontograph.synthesizer.self_check import (
    attach_self_check,
    format_self_check_report,
    run_self_check,
)

__all__ = [
    # generator
    "generate",
    "label_from_iri",
    "build_anchor_map",
    "SectionDraft",
    "ParagraphDraft",
    # self_check
    "run_self_check",
    "attach_self_check",
    "format_self_check_report",
]
