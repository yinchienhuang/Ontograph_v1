"""
ontograph/generator/instance_gen.py — LLM-based aerospace system instance generation.

One provider.complete() call generates a single top-level system with all its
subsystems and components.  The instance_count parameter controls roughly how many
total component individuals to aim for across the system (the LLM distributes them).

Two prompt modes:
  - Predefined domain (cubesat / uam / rocket / lunar): focused vocabulary with
    explicit system classes and required subsystems.
  - Custom domain (any other string): full taxonomy vocabulary; the LLM chooses
    the most appropriate classes for the given domain name.

Naming convention: component instances use realistic product/model designations
(e.g. OBC_Q7A, ICM42688P_1, NanoPower_P60) rather than type-revealing descriptive
names (e.g. OnboardComputer_Artemis).
"""

from __future__ import annotations

from ontograph.generator.schema import GeneratedSystem, GeneratedSystemBundle
from ontograph.generator.taxonomy import AerospaceTaxonomy, PREDEFINED_DOMAINS
from ontograph.llm.base import LLMMessage, LLMProvider, LLMRequest


# ── Internal helpers ──────────────────────────────────────────────────────────

def _has_ancestor(taxonomy: AerospaceTaxonomy, local: str, ancestor: str) -> bool:
    """Return True if class `local` has `ancestor` anywhere in its parent chain."""
    cls = taxonomy.get_class(local)
    while cls is not None and cls.parent is not None:
        if cls.parent == ancestor:
            return True
        cls = taxonomy.get_class(cls.parent)
    return False


# Abstract subsystem classes excluded from the "concrete subsystem" choice list —
# the LLM should pick specific subsystems, not these abstract roots.
_ABSTRACT_SUBSYSTEM_PARENTS = {"Subsystem", "PropulsionSubsystem"}

# Abstract system-level classes excluded from the top-level system choice list.
_ABSTRACT_SYSTEM_PARENTS = {
    "AerospaceSystem",
    "SpacecraftSystem",
    "AirVehicle",
    "Satellite",
    "LaunchVehicle",
    "LunarExplorationSystem",
    "UrbanAirMobility",
}


def _concrete_components(taxonomy: AerospaceTaxonomy, domain: str | None = None) -> list[str]:
    """Return local names of component leaf classes (is_component=True), optionally filtered."""
    return sorted(
        c.local for c in taxonomy.classes
        if c.is_component and (not domain or domain in c.domains)
    )


def _concrete_subsystems(taxonomy: AerospaceTaxonomy) -> list[str]:
    """Return local names of concrete (instantiable) subsystem classes."""
    return sorted(
        c.local for c in taxonomy.classes
        if not c.is_component
        and c.local not in _ABSTRACT_SUBSYSTEM_PARENTS
        and _has_ancestor(taxonomy, c.local, "Subsystem")
    )


def _concrete_systems(taxonomy: AerospaceTaxonomy) -> list[str]:
    """Return local names of concrete (non-abstract) top-level system classes."""
    return sorted(
        c.local for c in taxonomy.classes
        if not c.is_component
        and c.local not in _ABSTRACT_SYSTEM_PARENTS
        and _has_ancestor(taxonomy, c.local, "AerospaceSystem")
    )


def _props_text(taxonomy: AerospaceTaxonomy) -> str:
    """Format data properties as 'local (xsd_type, unit)' for prompt injection."""
    parts = []
    for dp in taxonomy.data_properties:
        unit = dp.unit or "N/A"
        parts.append(f"{dp.local} ({dp.xsd_type}, {unit})")
    return ", ".join(parts)


# ── Prompt builders ───────────────────────────────────────────────────────────

_NAMING_RULES = (
    "NAMING CONVENTION:\n"
    "  - local_name must be IRI-safe: only letters, digits, and underscores.\n"
    "  - Use realistic product/model designations — NOT type-revealing descriptive names.\n"
    "    SYSTEMS:    DOVE_Alpha3, HawkEye_SAT2, Falcon9_SN15\n"
    "    SUBSYSTEMS: EPS_NP60, ADCS_Hyp200, CDH_iOBC\n"
    "    COMPONENTS: ICM42688P_1, OBC_Q7A, NanoPower_P60, RWP050_2, STT200_A\n"
    "  - AVOID names like OnboardComputer_Artemis or StarTracker_Mission1.\n"
    "  - For redundant units append _1, _2 after the model designation.\n"
)


def _instance_count_constraint(instance_count: int) -> str:
    return (
        f"INSTANCE COUNT:\n"
        f"  - Target approximately {instance_count} total component instances across all subsystems.\n"
        f"  - Distribute components realistically across subsystems.\n"
        f"  - Where redundancy is physically meaningful (e.g. sensors, actuators, panels),\n"
        f"    create 2–3 instances of the same component type with numeric suffixes.\n"
        f"    Avoid redundancy where it is unrealistic.\n"
        f"  - Provide 3–6 data property attributes per entity.\n"
        f"  - Attribute values must be physically realistic for this domain.\n"
        f"  - Use 'decimal' datatype for numeric values, 'string' for text.\n"
        f"  - All labels and comments must be specific, not generic placeholders.\n"
        f"  - Do not invent class names — use only the classes listed above.\n"
    )


def _build_predefined_prompt(
    domain: str,
    taxonomy: AerospaceTaxonomy,
    instance_count: int,
) -> str:
    sys_classes = taxonomy.domain_system_classes[domain]
    req_subs    = taxonomy.domain_required_subsystems[domain]
    comp_names  = _concrete_components(taxonomy, domain)
    props       = _props_text(taxonomy)

    return (
        f"You are an aerospace systems engineer generating realistic OWL ontology "
        f"instances for a single '{domain}' system design.\n\n"
        f"TOP-LEVEL SYSTEM CLASS — choose exactly one:\n"
        + "".join(f"  {c}\n" for c in sys_classes)
        + f"\nREQUIRED SUBSYSTEM CLASSES — include ALL of these:\n"
        + "".join(f"  {s}\n" for s in req_subs)
        + f"\nAVAILABLE COMPONENT CLASSES — choose appropriate classes per subsystem:\n"
        f"  {', '.join(comp_names)}\n\n"
        f"AVAILABLE DATA PROPERTIES:\n"
        f"  {props}\n\n"
        + _NAMING_RULES
        + _instance_count_constraint(instance_count)
    )


def _build_custom_prompt(
    domain: str,
    taxonomy: AerospaceTaxonomy,
    instance_count: int,
) -> str:
    sys_names  = _concrete_systems(taxonomy)
    sub_names  = _concrete_subsystems(taxonomy)
    comp_names = _concrete_components(taxonomy)  # no domain filter
    props      = _props_text(taxonomy)

    return (
        f"You are an aerospace systems engineer generating realistic OWL ontology "
        f"instances for a single '{domain}' system design.\n\n"
        f"This is a CUSTOM DOMAIN. Choose the most appropriate classes from the "
        f"taxonomy below that best represent a real-world '{domain}' system.\n\n"
        f"AVAILABLE SYSTEM CLASSES (choose the best match for '{domain}'):\n"
        f"  {', '.join(sys_names)}\n\n"
        f"AVAILABLE SUBSYSTEM CLASSES — choose 4–6 relevant ones:\n"
        f"  {', '.join(sub_names)}\n\n"
        f"AVAILABLE COMPONENT CLASSES — choose appropriate classes per subsystem:\n"
        f"  {', '.join(comp_names)}\n\n"
        f"AVAILABLE DATA PROPERTIES:\n"
        f"  {props}\n\n"
        + _NAMING_RULES
        + _instance_count_constraint(instance_count)
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_system(
    domain: str,
    taxonomy: AerospaceTaxonomy,
    provider: LLMProvider,
    namespace: str,
    instance_count: int = 15,
    temperature: float = 0.7,
) -> GeneratedSystem:
    """
    Generate one top-level aerospace system with its subsystems and components.

    Makes a single provider.complete() call and returns the parsed GeneratedSystem.

    Args:
        domain:         Domain string (predefined or custom).
        taxonomy:       The AEROSPACE_TAXONOMY singleton.
        provider:       LLM provider instance.
        namespace:      Ontology namespace IRI (informational only in this call).
        instance_count: Target total number of component instances across the system.
        temperature:    LLM temperature (default 0.7 for variety).
    """
    if domain in PREDEFINED_DOMAINS:
        system_content = _build_predefined_prompt(domain, taxonomy, instance_count)
    else:
        system_content = _build_custom_prompt(domain, taxonomy, instance_count)

    user_content = (
        f"Generate a single '{domain}' system design with a unique, specific mission "
        f"or vehicle designation (use a realistic model/program name, not a generic label). "
        f"Create the full system hierarchy: required subsystems each populated with "
        f"components targeting approximately {instance_count} total component instances. "
        f"Use realistic product model names for all local_name values — see naming convention. "
        f"All attribute values must be realistic and specific to this domain."
    )

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user",   content=user_content),
        ],
        response_model=GeneratedSystemBundle,
        temperature=temperature,
        max_tokens=8192,
    )

    response = provider.complete(request)
    return response.parsed.system
