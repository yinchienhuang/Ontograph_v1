"""
ontograph/generator/owl_builder.py — Build and serialize the aerospace OWL graph.

Assembles a complete rdflib Graph containing:
  - TBox: all ClassDef (owl:Class + rdfs:subClassOf), DataPropDef
          (owl:DatatypeProperty), and ObjectPropDef (owl:ObjectProperty)
  - ABox: all GeneratedSystem / GeneratedSubsystem / GeneratedComponent individuals
          as owl:NamedIndividual triples with data and object properties

The output is serialized as RDF/XML (.owl) using the existing save_graph() helper.
"""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from ontograph.generator.schema import (
    GeneratedAttribute,
    GeneratedComponent,
    GeneratedSubsystem,
    GeneratedSystem,
)
from ontograph.generator.taxonomy import AerospaceTaxonomy
from ontograph.utils.owl import empty_graph, save_graph

# Map from schema datatype strings → rdflib XSD URIRefs
_XSD_MAP = {
    "decimal": XSD.decimal,
    "integer": XSD.integer,
    "string":  XSD.string,
    "boolean": XSD.boolean,
}


# ── Public API ────────────────────────────────────────────────────────────────

def build_owl_graph(
    systems: list[GeneratedSystem],
    taxonomy: AerospaceTaxonomy,
    namespace: str,
) -> Graph:
    """
    Build a complete rdflib Graph with TBox + ABox.

    Args:
        systems:   List of generated system individuals to include in the ABox.
        taxonomy:  The AEROSPACE_TAXONOMY singleton (provides TBox definitions).
        namespace: Ontology namespace IRI (e.g. 'http://example.org/aerospace#').

    Returns:
        rdflib Graph with all TBox declarations and ABox individuals.
    """
    NS = Namespace(namespace)
    g = empty_graph(("aerospace", namespace))

    # ── OWL Ontology header ───────────────────────────────────────────────────
    ont_iri = namespace.rstrip("#/")
    g.add((URIRef(ont_iri), RDF.type, OWL.Ontology))

    # ── TBox: Classes ─────────────────────────────────────────────────────────
    for cls in taxonomy.classes:
        uri = NS[cls.local]
        g.add((uri, RDF.type, OWL.Class))
        if cls.parent:
            g.add((uri, RDFS.subClassOf, NS[cls.parent]))
        g.add((uri, RDFS.label,   Literal(cls.label)))
        g.add((uri, RDFS.comment, Literal(cls.description)))

    # ── TBox: Data Properties ─────────────────────────────────────────────────
    for dp in taxonomy.data_properties:
        uri = NS[dp.local]
        g.add((uri, RDF.type,      OWL.DatatypeProperty))
        g.add((uri, RDFS.label,    Literal(dp.label)))
        g.add((uri, RDFS.comment,  Literal(dp.description)))
        g.add((uri, RDFS.range,    XSD[dp.xsd_type]))

    # ── TBox: Object Properties ───────────────────────────────────────────────
    for op in taxonomy.object_properties:
        uri = NS[op.local]
        g.add((uri, RDF.type,      OWL.ObjectProperty))
        g.add((uri, RDFS.label,    Literal(op.label)))
        g.add((uri, RDFS.comment,  Literal(op.description)))
        g.add((uri, RDFS.domain,   NS[op.domain_class]))
        g.add((uri, RDFS.range,    NS[op.range_class]))

    # ── ABox ──────────────────────────────────────────────────────────────────
    for sys in systems:
        _add_system(g, NS, sys)

    return g


def serialize_owl(graph: Graph, output_path: Path) -> None:
    """Write the graph to an RDF/XML .owl file."""
    save_graph(graph, output_path, fmt="xml")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_attr(g: Graph, NS: Namespace, subj: URIRef, attr: GeneratedAttribute) -> None:
    dt = _XSD_MAP.get(attr.datatype, XSD.string)
    g.add((subj, NS[attr.property_local], Literal(attr.value, datatype=dt)))


def _add_component(g: Graph, NS: Namespace, comp: GeneratedComponent) -> URIRef:
    uri = NS[comp.local_name]
    g.add((uri, RDF.type,      OWL.NamedIndividual))
    g.add((uri, RDF.type,      NS[comp.class_local]))
    g.add((uri, RDFS.label,    Literal(comp.label)))
    g.add((uri, RDFS.comment,  Literal(comp.comment)))
    for attr in comp.attributes:
        _add_attr(g, NS, uri, attr)
    return uri


def _add_subsystem(g: Graph, NS: Namespace, sub: GeneratedSubsystem) -> URIRef:
    uri = NS[sub.local_name]
    g.add((uri, RDF.type,      OWL.NamedIndividual))
    g.add((uri, RDF.type,      NS[sub.class_local]))
    g.add((uri, RDFS.label,    Literal(sub.label)))
    g.add((uri, RDFS.comment,  Literal(sub.comment)))
    for attr in sub.attributes:
        _add_attr(g, NS, uri, attr)
    for comp in sub.components:
        comp_uri = _add_component(g, NS, comp)
        g.add((uri, NS["hasComponent"], comp_uri))
    return uri


def _add_system(g: Graph, NS: Namespace, sys: GeneratedSystem) -> URIRef:
    uri = NS[sys.local_name]
    g.add((uri, RDF.type,      OWL.NamedIndividual))
    g.add((uri, RDF.type,      NS[sys.class_local]))
    g.add((uri, RDFS.label,    Literal(sys.label)))
    g.add((uri, RDFS.comment,  Literal(sys.comment)))
    for attr in sys.attributes:
        _add_attr(g, NS, uri, attr)
    for sub in sys.subsystems:
        sub_uri = _add_subsystem(g, NS, sub)
        g.add((uri, NS["hasSubsystem"], sub_uri))
    return uri
