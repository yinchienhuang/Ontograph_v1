"""
ontograph/generator — Aerospace OWL ontology generator (standalone module).

Generates a complete aerospace OWL ontology from scratch:
  - TBox: hardcoded class hierarchy + properties (taxonomy.py)
  - ABox: LLM-generated instances with realistic attributes (instance_gen.py)
  - Output: serialized RDF/XML .owl file (owl_builder.py)

Standalone — does not interact with the ingest/synthesizer/evaluator pipeline.

Usage:
    python scripts/generate_ontology.py --domain cubesat --count 15 --provider claude
"""

from ontograph.generator.taxonomy import (
    AerospaceTaxonomy,
    ClassDef,
    DataPropDef,
    ObjectPropDef,
    AEROSPACE_TAXONOMY,
    PREDEFINED_DOMAINS,
)
from ontograph.generator.schema import (
    GeneratedAttribute,
    GeneratedComponent,
    GeneratedSubsystem,
    GeneratedSystem,
    GeneratedSystemBundle,
)
from ontograph.generator.instance_gen import generate_system
from ontograph.generator.owl_builder import build_owl_graph, serialize_owl

__all__ = [
    # Taxonomy
    "AerospaceTaxonomy",
    "ClassDef",
    "DataPropDef",
    "ObjectPropDef",
    "AEROSPACE_TAXONOMY",
    "PREDEFINED_DOMAINS",
    # Schema
    "GeneratedAttribute",
    "GeneratedComponent",
    "GeneratedSubsystem",
    "GeneratedSystem",
    "GeneratedSystemBundle",
    # Generation
    "generate_system",
    # OWL building
    "build_owl_graph",
    "serialize_owl",
]
