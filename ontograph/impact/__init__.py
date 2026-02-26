"""impact — Design-change impact analysis module."""

from ontograph.impact.schema import (
    AttributeChangeSpec,
    ImpactArmResult,
    ImpactAnalysisResult,
    ImpactScenario,
)
from ontograph.impact.loader import load_scenarios
from ontograph.impact.analyzer import analyze_impact

__all__ = [
    "AttributeChangeSpec",
    "ImpactScenario",
    "ImpactArmResult",
    "ImpactAnalysisResult",
    "load_scenarios",
    "analyze_impact",
]
