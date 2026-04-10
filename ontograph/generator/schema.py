"""
ontograph/generator/schema.py — Pydantic models for LLM-generated OWL instances.

The LLM returns a GeneratedSystemBundle for each top-level system it creates.
All local_name values must be IRI-safe (letters, digits, underscores only — no spaces
or special characters). The LLM is instructed to follow this convention in the prompt.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedAttribute(BaseModel):
    """One data property triple: property_local → value."""

    property_local: str
    "Local name of the data property (must match a DataPropDef.local from the taxonomy)."

    value: str
    "String representation of the attribute value."

    datatype: str = "decimal"
    "XSD datatype: 'decimal' | 'integer' | 'string'."


class GeneratedComponent(BaseModel):
    """One component individual with its data property attributes."""

    local_name: str
    "IRI-safe identifier, e.g. 'SolarPanel_Phoenix_01'."

    class_local: str
    "Local name of a Component subclass from the taxonomy."

    label: str
    "Human-readable label for this component."

    comment: str
    "Brief, specific description of this component's role and specs."

    attributes: list[GeneratedAttribute] = Field(default_factory=list)


class GeneratedSubsystem(BaseModel):
    """One subsystem individual with its components and data property attributes."""

    local_name: str

    class_local: str
    "Local name of a Subsystem subclass from the taxonomy."

    label: str
    comment: str

    attributes: list[GeneratedAttribute] = Field(default_factory=list)
    components: list[GeneratedComponent] = Field(default_factory=list)


class GeneratedSystem(BaseModel):
    """One top-level system individual (CubeSat, rocket, eVTOL, etc.)."""

    local_name: str
    "IRI-safe identifier, e.g. 'CubeSat_Artemis_1'."

    class_local: str
    "Local name of a system class from the taxonomy (e.g. 'NanoSatellite')."

    label: str
    comment: str

    attributes: list[GeneratedAttribute] = Field(default_factory=list)
    subsystems: list[GeneratedSubsystem] = Field(default_factory=list)


class GeneratedSystemBundle(BaseModel):
    """LLM response wrapper: one top-level system with its full hierarchy."""

    system: GeneratedSystem
