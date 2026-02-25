"""
utils/owl.py — rdflib helpers for reading and writing the working ontology.

All OWL manipulation goes through this module so the rest of the codebase
never imports rdflib directly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

from ontograph.models.ontology import (
    ChangeSource,
    OntologyDelta,
    OntologyDeltaEntry,
    OntologyTriple,
)

# ---------------------------------------------------------------------------
# Well-known namespaces re-exported so callers don't need to import rdflib
# ---------------------------------------------------------------------------

__all__ = [
    "load_graph", "save_graph", "graph_sha256",
    "add_entry", "add_entries", "remove_triple",
    "sparql_query", "iter_triples",
    "iri", "literal",
    "copy_tbox",
    "owl_to_delta",
    "read_tbox_summary",
    "TBoxSummary",
    "OWL", "RDF", "RDFS", "SKOS", "XSD",
]


# ---------------------------------------------------------------------------
# Schema meta-types — subjects with these rdf:type values are TBox axioms,
# not ABox individuals.  Used by both copy_tbox() and owl_to_delta().
# ---------------------------------------------------------------------------

_SCHEMA_TYPES: frozenset[URIRef] = frozenset({
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    OWL.FunctionalProperty,
    OWL.InverseFunctionalProperty,
    OWL.TransitiveProperty,
    OWL.SymmetricProperty,
    OWL.AsymmetricProperty,
    OWL.ReflexiveProperty,
    OWL.IrreflexiveProperty,
    OWL.Ontology,
    OWL.Restriction,
    OWL.AllDisjointClasses,
    OWL.AllDisjointProperties,
    RDF.Property,
    RDFS.Class,
    RDFS.Datatype,
})


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def load_graph(path: str | Path, fmt: str = "turtle") -> Graph:
    """Load an OWL/RDF file into an rdflib Graph."""
    g = Graph()
    g.parse(str(path), format=fmt)
    return g


def save_graph(graph: Graph, path: str | Path, fmt: str = "turtle") -> None:
    """Serialize an rdflib Graph to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(path), format=fmt)


def graph_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the OWL file bytes."""
    from ontograph.utils.io import sha256_file
    return sha256_file(path)


def copy_tbox(
    tbox_path: str | Path,
    graph: Graph,
    fmt: str = "xml",
) -> int:
    """
    Merge TBox-only triples from *tbox_path* into *graph*.

    **ABox individuals are stripped**: any subject declared as
    ``owl:NamedIndividual``, or whose ``rdf:type`` points to a user-defined
    class (rather than a schema meta-type like ``owl:Class``), is excluded
    along with all of its associated triples.

    What IS preserved:
      - ``owl:Class`` declarations
      - ``rdfs:subClassOf`` axioms (full class hierarchy)
      - ``owl:ObjectProperty`` / ``owl:DatatypeProperty`` declarations
      - ``rdfs:domain`` and ``rdfs:range`` restrictions
      - ``rdfs:label`` and ``rdfs:comment`` on classes/properties
      - ``owl:Ontology`` header triples

    Because rdflib graphs are sets, re-merging the same TBox into an existing
    graph is idempotent.

    Args:
        tbox_path: Path to the source TBox OWL file.
        graph:     Target rdflib Graph to merge into.
        fmt:       rdflib format string (``"xml"``, ``"turtle"``, etc.).

    Returns:
        Number of triples copied into *graph*.
    """
    # Parse into a scratch graph so we can filter before merging
    tmp = Graph()
    tmp.parse(str(tbox_path), format=fmt)

    # Step 1: identify every subject that is a schema axiom carrier
    schema_subjects: set[URIRef] = set()
    for schema_type in _SCHEMA_TYPES:
        for s in tmp.subjects(RDF.type, schema_type):
            if not isinstance(s, BNode):
                schema_subjects.add(s)  # type: ignore[arg-type]

    # Step 2: identify individuals — anything that has rdf:type pointing to a
    # user class (not a schema meta-type) and is NOT itself a schema subject.
    individuals: set[URIRef] = set()
    for s in tmp.subjects(RDF.type, OWL.NamedIndividual):
        if not isinstance(s, BNode):
            individuals.add(s)  # type: ignore[arg-type]
    for s, _p, o in tmp.triples((None, RDF.type, None)):
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue
        if s in schema_subjects:
            continue
        if o not in _SCHEMA_TYPES:
            individuals.add(s)  # type: ignore[arg-type]

    # Step 3: copy only non-individual triples
    before = len(graph)
    for s, p, o in tmp:
        if s in individuals:
            continue
        # Also skip triples where the individual appears as object
        # (e.g. owl:hasValue pointing to an instance) — rare but clean
        if isinstance(o, URIRef) and o in individuals:
            continue
        graph.add((s, p, o))

    return len(graph) - before


def empty_graph(*namespaces: tuple[str, str]) -> Graph:
    """
    Create an empty rdflib Graph with common namespaces bound.

    Pass additional (prefix, uri) pairs to bind domain-specific prefixes.
    """
    g = Graph()
    g.bind("owl",  OWL)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("xsd",  XSD)
    for prefix, uri in namespaces:
        g.bind(prefix, Namespace(uri))
    return g


# ---------------------------------------------------------------------------
# Triple helpers
# ---------------------------------------------------------------------------

def iri(value: str) -> URIRef:
    """Wrap a string IRI as an rdflib URIRef."""
    return URIRef(value)


def literal(value: str, datatype: str | None = None, lang: str | None = None) -> Literal:
    """Create an rdflib Literal, optionally typed or tagged."""
    if datatype:
        return Literal(value, datatype=URIRef(datatype))
    if lang:
        return Literal(value, lang=lang)
    return Literal(value)


def _triple_to_rdflib(
    triple: OntologyTriple,
) -> tuple[URIRef, URIRef, URIRef | Literal]:
    """Convert an OntologyTriple to an rdflib (s, p, o) tuple."""
    s = URIRef(triple.subject)
    p = URIRef(triple.predicate)
    if triple.datatype or triple.language:
        o: URIRef | Literal = literal(triple.object, triple.datatype, triple.language)
    else:
        # Heuristic: if object starts with http or a known prefix, treat as IRI
        o = URIRef(triple.object) if triple.object.startswith("http") else literal(triple.object)
    return s, p, o


# ---------------------------------------------------------------------------
# Adding / removing triples
# ---------------------------------------------------------------------------

def add_entry(graph: Graph, entry: OntologyDeltaEntry) -> None:
    """Add an approved OntologyDeltaEntry's triple to the graph."""
    s, p, o = _triple_to_rdflib(entry.triple)
    graph.add((s, p, o))


def add_entries(graph: Graph, entries: list[OntologyDeltaEntry]) -> None:
    """Bulk-add a list of approved entries."""
    for entry in entries:
        add_entry(graph, entry)


def remove_triple(graph: Graph, triple: OntologyTriple) -> None:
    """Remove one triple from the graph (no-op if not present)."""
    s, p, o = _triple_to_rdflib(triple)
    graph.remove((s, p, o))


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def sparql_query(graph: Graph, query: str) -> list[dict[str, str]]:
    """
    Execute a SPARQL SELECT query and return results as a list of dicts.

    Each dict maps variable name → string value (IRI or literal).
    """
    results = []
    for row in graph.query(query):
        results.append({str(var): str(row[var]) for var in row.labels})
    return results


def iter_triples(graph: Graph) -> Iterator[tuple[str, str, str]]:
    """Yield (subject, predicate, object) as plain strings."""
    for s, p, o in graph:
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue  # skip blank nodes
        yield str(s), str(p), str(o)


def neighbors_one_hop(graph: Graph, subject_iri: str) -> list[dict[str, str]]:
    """
    Return all triples where `subject_iri` appears as subject or object.

    Used by the evaluator's docs+ontology arm to augment retrieval context.
    """
    uri = URIRef(subject_iri)
    results: list[dict[str, str]] = []
    for s, p, o in graph.triples((uri, None, None)):
        results.append({"subject": str(s), "predicate": str(p), "object": str(o)})
    for s, p, o in graph.triples((None, None, uri)):
        results.append({"subject": str(s), "predicate": str(p), "object": str(o)})
    return results


# ---------------------------------------------------------------------------
# TBox introspection
# ---------------------------------------------------------------------------

@dataclass
class DatatypePropertyInfo:
    """Local name + XSD range for one datatype property."""
    local_name: str
    range_xsd: str | None  # e.g. "decimal", "integer", "string"


@dataclass
class TBoxSummary:
    """
    A lightweight snapshot of the TBox vocabulary in an OWL file.

    Suitable for injecting into an LLM prompt so it maps extracted entities
    to existing class and property names rather than inventing new ones.
    """
    namespace: str                          # primary namespace IRI
    classes: list[str] = field(default_factory=list)          # local names
    object_properties: list[str] = field(default_factory=list)  # local names
    datatype_properties: list[DatatypePropertyInfo] = field(default_factory=list)

    def classes_text(self) -> str:
        """Comma-separated class local names for prompt injection."""
        return ", ".join(sorted(self.classes))

    def object_props_text(self) -> str:
        """Comma-separated object property local names for prompt injection."""
        return ", ".join(sorted(op for op in self.object_properties))

    def datatype_props_text(self) -> str:
        """
        Formatted list of datatype properties for prompt injection.
        Example: "massKg (xsd:decimal), heritageTRL (xsd:integer)"
        """
        parts = []
        for dp in sorted(self.datatype_properties, key=lambda x: x.local_name):
            if dp.range_xsd:
                parts.append(f"{dp.local_name} (xsd:{dp.range_xsd})")
            else:
                parts.append(dp.local_name)
        return ", ".join(parts)

    def to_prompt_block(self) -> str:
        """
        Multi-line string ready to paste into an LLM system prompt.

        Instructs the model to prefer TBox vocabulary over invented names.
        """
        lines = [
            f"ONTOLOGY NAMESPACE: {self.namespace}",
            "",
            "AVAILABLE CLASSES (use for rdf_type — prefer these over generic names):",
            f"  {self.classes_text()}",
            "",
            "AVAILABLE DATATYPE PROPERTIES (use for attribute predicates):",
            f"  {self.datatype_props_text()}",
        ]
        if self.object_properties:
            lines += [
                "",
                "AVAILABLE OBJECT PROPERTIES (use when object is another entity/IRI):",
                f"  {self.object_props_text()}",
            ]
        return "\n".join(lines)


def read_tbox_summary(path: str | Path, fmt: str = "xml") -> TBoxSummary:
    """
    Parse an OWL TBox file and return a :class:`TBoxSummary`.

    The function detects the primary namespace (the most common non-standard
    namespace among declared classes), then collects:
      - OWL named classes
      - OWL object properties
      - OWL datatype properties (with their XSD range if declared)

    Args:
        path: Path to the OWL file.
        fmt:  rdflib format string — ``"xml"`` for RDF/XML (default),
              ``"turtle"`` for Turtle, ``"n3"``, etc.

    Returns:
        A :class:`TBoxSummary` with vocabulary extracted from the TBox.
    """
    g = load_graph(path, fmt=fmt)

    _SKIP_NS = {
        str(OWL), str(RDF), str(RDFS), str(XSD), str(SKOS),
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2001/XMLSchema#",
    }

    def _local(uri: str, ns: str) -> str | None:
        """Return local name if uri starts with ns, else None."""
        if uri.startswith(ns):
            return uri[len(ns):]
        return None

    # --- Detect primary namespace -------------------------------------------
    # Collect all subjects that are OWL Classes or OWL Properties
    ns_counts: dict[str, int] = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        uri = str(cls)
        if uri.startswith("http") and "#" in uri:
            ns = uri.rsplit("#", 1)[0] + "#"
            if ns not in _SKIP_NS:
                ns_counts[ns] = ns_counts.get(ns, 0) + 1

    if ns_counts:
        namespace = max(ns_counts, key=lambda k: ns_counts[k])
    else:
        # Fall back to the first declared namespace that isn't a W3C one
        namespace = next(
            (str(ns) for _prefix, ns in g.namespaces()
             if str(ns) not in _SKIP_NS and str(ns).startswith("http")),
            "http://example.org/#",
        )

    # --- Classes ------------------------------------------------------------
    classes: list[str] = []
    for cls in g.subjects(RDF.type, OWL.Class):
        uri = str(cls)
        if isinstance(cls, BNode):
            continue
        loc = _local(uri, namespace)
        if loc and loc.strip():
            classes.append(loc)

    # --- Object properties --------------------------------------------------
    object_properties: list[str] = []
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        uri = str(prop)
        loc = _local(uri, namespace)
        if loc and loc.strip():
            object_properties.append(loc)

    # --- Datatype properties ------------------------------------------------
    datatype_properties: list[DatatypePropertyInfo] = []
    _XSD_STR = str(XSD)
    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        uri = str(prop)
        loc = _local(uri, namespace)
        if not loc or not loc.strip():
            continue
        # Look for rdfs:range → XSD type
        range_xsd: str | None = None
        for range_node in g.objects(prop, RDFS.range):
            r = str(range_node)
            if r.startswith(_XSD_STR):
                range_xsd = r[len(_XSD_STR):]  # e.g. "decimal", "integer"
                break
        datatype_properties.append(
            DatatypePropertyInfo(local_name=loc, range_xsd=range_xsd)
        )

    return TBoxSummary(
        namespace=namespace,
        classes=sorted(classes),
        object_properties=sorted(object_properties),
        datatype_properties=sorted(datatype_properties, key=lambda x: x.local_name),
    )


def owl_to_delta(
    graph: Graph,
    delta_id: str,
    base_iri: str,
    created_at: str | None = None,
) -> OntologyDelta:
    """
    Convert ABox individuals from an rdflib Graph into an OntologyDelta.

    Every triple whose subject is an ABox individual (not a TBox class or
    property declaration) is converted to an ``OntologyDeltaEntry`` with
    ``status="approved"`` so the result can be passed directly to
    :func:`~ontograph.synthesizer.generator.generate`.

    Blank-node subjects and blank-node objects are silently skipped.

    Args:
        graph:      rdflib Graph loaded from an OWL file.
        delta_id:   Identifier for the returned ``OntologyDelta``
                    (e.g. the OWL file stem).
        base_iri:   The primary namespace IRI of the ontology.
        created_at: ISO 8601 timestamp.  Defaults to the current UTC time.

    Returns:
        An ``OntologyDelta`` whose entries are all ``approved``.
    """
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    # Identify schema subjects (TBox axiom carriers)
    schema_subjects: set[URIRef] = set()
    for schema_type in _SCHEMA_TYPES:
        for s in graph.subjects(RDF.type, schema_type):
            if not isinstance(s, BNode):
                schema_subjects.add(s)  # type: ignore[arg-type]

    # Identify ABox individuals
    individuals: set[URIRef] = set()
    for s in graph.subjects(RDF.type, OWL.NamedIndividual):
        if not isinstance(s, BNode):
            individuals.add(s)  # type: ignore[arg-type]
    for s, _p, o in graph.triples((None, RDF.type, None)):
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue
        if s in schema_subjects:
            continue
        if o not in _SCHEMA_TYPES:
            individuals.add(s)  # type: ignore[arg-type]

    # Convert each individual's triples to OntologyDeltaEntry objects
    entries: list[OntologyDeltaEntry] = []
    for ind in sorted(individuals, key=str):  # sorted for stable ordering
        for s, p, o in graph.triples((ind, None, None)):
            if isinstance(o, BNode):
                continue

            if isinstance(o, Literal):
                triple = OntologyTriple(
                    subject=str(s),
                    predicate=str(p),
                    object=str(o),
                    datatype=str(o.datatype) if o.datatype else None,
                    language=o.language or None,
                )
            else:  # URIRef
                triple = OntologyTriple(
                    subject=str(s),
                    predicate=str(p),
                    object=str(o),
                )

            entry_id = hashlib.sha256(
                f"{s}|{p}|{o}".encode()
            ).hexdigest()[:16]

            entries.append(OntologyDeltaEntry(
                id=entry_id,
                triple=triple,
                rationale="Extracted from OWL ABox",
                confidence=1.0,
                change_source=ChangeSource.PIPELINE,
                status="approved",
            ))

    return OntologyDelta(
        id=delta_id,
        extraction_bundle_id=delta_id + "-bundle",
        base_ontology_iri=base_iri,
        created_at=created_at,
        entries=entries,
    )
