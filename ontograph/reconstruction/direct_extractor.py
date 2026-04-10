"""
ontograph/reconstruction/direct_extractor.py — General-AI arm: single LLM call.

Pours the full raw document text into one LLM call together with the TBox
vocabulary (class names + property names) from the source ontology.
No individual names are provided — the LLM must discover them from the text,
just as the onto-graph arm does.  No chunking, no section tracking.

This is the fairest possible baseline: the LLM has the same TBox vocabulary
as the onto-graph pipeline, but none of the structured extraction machinery
(chunking, entity mapping, alignment).
"""

from __future__ import annotations

from ontograph.llm.base import LLMMessage, LLMProvider, LLMRequest
from ontograph.reconstruction.schema import DirectExtractionResult


def extract_direct(
    document_text: str,
    namespace: str,
    class_locals: list[str],
    property_locals: list[str],
    provider: LLMProvider,
    temperature: float = 0.0,
) -> DirectExtractionResult:
    """
    Ask the LLM to extract all individuals and their property assertions
    directly from *document_text* using the TBox vocabulary as a guide.

    Makes exactly one provider.complete() call.

    Args:
        document_text:   Full raw document text (Markdown or plain text).
        namespace:       Ontology namespace IRI (informational only).
        class_locals:    Known class local names from the TBox (e.g. "OnboardComputer").
        property_locals: Known property local names from the TBox (e.g. "massKg").
        provider:        LLM provider instance.
        temperature:     LLM temperature (default 0.0 for deterministic extraction).

    Returns:
        :class:`~ontograph.reconstruction.schema.DirectExtractionResult` with
        all extracted triples including ``rdf_type`` class assignments.
    """
    classes_block    = "\n".join(f"  {name}" for name in sorted(class_locals))
    properties_block = "\n".join(f"  {name}" for name in sorted(property_locals))

    system_content = (
        "You are a knowledge extraction system. Read the document and extract all "
        "individuals (named entities) together with their property values.\n\n"
        f"ONTOLOGY NAMESPACE: {namespace}\n\n"
        "AVAILABLE CLASSES (assign the best match as rdf_type for each individual):\n"
        f"{classes_block}\n\n"
        "AVAILABLE PROPERTIES (use exact names):\n"
        f"{properties_block}\n\n"
        "OUTPUT RULES:\n"
        "  - Discover individual names from the document text — do NOT invent entities.\n"
        "  - subject must be an IRI-safe local name (letters, digits, underscores only).\n"
        "  - rdf_type must be one of the class names listed above, or null if none fits.\n"
        "  - property must be one of the property names listed above.\n"
        "  - value must be a plain string (numeric or text) without units.\n"
        "  - Emit one triple per (subject, property) fact stated in the document.\n"
        "  - If a fact is not explicitly stated, do not include it.\n"
        "  - Numeric values as plain numbers (e.g. '1.5', not '1.5 kg').\n"
    )

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user",   content=document_text),
        ],
        response_model=DirectExtractionResult,
        temperature=temperature,
        max_tokens=8192,
    )

    response = provider.complete(request)
    return response.parsed
