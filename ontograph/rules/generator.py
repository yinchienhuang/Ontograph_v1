"""
rules/generator.py — LLM-based plain-English rule description generator.

RESEARCH PURPOSE:
    The generated text deliberately simulates *vague engineering documentation* —
    the kind of prose an engineer might write without access to exact thresholds or
    precise property names.  This creates the document-mode baseline for the research
    comparison: how well does an LLM detect violations from vague documentation vs.
    from precise structured ontology data?

    The generator is intentionally instructed to OMIT:
      - Exact numeric thresholds
      - Specific property/attribute names
      - Operator symbols (>, <, etc.)

    producing output like:
      "Thruster mounting hardware may have compatibility issues with heavier propellant
      tanks. Engineers should verify that the selected tank configuration is suitable
      for the propulsion subsystem."

    rather than a precise reformulation like:
      "A PropulsionSubsystem is not compatible with a PropellantTank whose hasDryMass
      exceeds 25.0 kg."
"""

from __future__ import annotations

from pydantic import BaseModel

from ontograph.llm import LLMMessage, LLMProvider, LLMRequest
from ontograph.rules.schema import OrgRule


# ---------------------------------------------------------------------------
# LLM response schema
# ---------------------------------------------------------------------------

class PlainEnglishOutput(BaseModel):
    """One vague, documentation-style sentence or two about the rule."""
    plain_english: str


# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to produce vague, incomplete prose
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are simulating the kind of brief compatibility note that might appear in an "
    "engineering document written without precise specifications. "
    "Given a structured rule, write 1–2 sentences that capture only the general concern — "
    "do NOT include exact numeric thresholds, specific property names, or operator symbols. "
    "Use vague qualifiers like 'heavy', 'large', 'may cause issues', 'should be checked'. "
    "The result should read like informal documentation written by an engineer who knows "
    "there is a concern but does not remember the exact values."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_plain_english(rule: OrgRule, provider: LLMProvider) -> str:
    """
    Generate a vague, documentation-style description for *rule*.

    The LLM is given only the rule name, note, and subject/object type names —
    NOT the exact threshold condition from ``rule.when``.  This deliberately
    produces incomplete prose suitable for the document-mode research baseline.

    Args:
        rule:     The structured rule to describe.
        provider: LLM provider to call.

    Returns:
        A 1–2 sentence string.  Suitable for storing in ``rule.plain_english``.
    """
    subject = rule.subject_type or "component"
    obj     = rule.object_type  or "another component"
    note_block = f"\nAdditional note: {rule.note}" if rule.note else ""

    user_msg = (
        f"Rule name: {rule.name}\n"
        f"Subject component type: {subject}\n"
        f"Object component type: {obj}"
        f"{note_block}\n\n"
        "Write a vague, general compatibility note for this rule. "
        "Do not mention exact values, thresholds, or attribute names."
    )

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user",   content=user_msg),
        ],
        response_model=PlainEnglishOutput,
    )
    response = provider.complete(request)
    return response.parsed.plain_english


def generate_all_plain_english(
    rules: list[OrgRule],
    provider: LLMProvider,
) -> list[OrgRule]:
    """
    Fill ``plain_english`` on every rule that does not already have one.

    Rules that already have a non-empty ``plain_english`` value are skipped.

    Args:
        rules:    List of OrgRule objects (as returned by load_rules).
        provider: LLM provider to call.

    Returns:
        A new list of OrgRule objects with ``plain_english`` populated on all.
    """
    updated: list[OrgRule] = []
    for rule in rules:
        if rule.plain_english.strip():
            updated.append(rule)
        else:
            text = generate_plain_english(rule, provider)
            updated.append(rule.model_copy(update={"plain_english": text}))
    return updated
