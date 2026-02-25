"""
Unit tests for utils/io.py — JSON persistence helpers.
"""

import json
from pathlib import Path

import pytest

from ontograph.models.ontology import (
    ChangeSource,
    OntologyDelta,
    OntologyDeltaEntry,
    OntologyTriple,
)
from ontograph.utils.io import (
    exists,
    list_ids,
    load,
    load_by_id,
    save,
    sha256_str,
)


def _make_delta(delta_id: str = "delta1") -> OntologyDelta:
    triple = OntologyTriple(
        subject="aero:Thruster_1",
        predicate="aero:hasMass",
        object="89",
        datatype="xsd:float",
    )
    entry = OntologyDeltaEntry(
        id="entry1", triple=triple,
        rationale="test", confidence=0.9,
        change_source=ChangeSource.PIPELINE,
    )
    return OntologyDelta(
        id=delta_id,
        extraction_bundle_id="bundle1",
        base_ontology_iri="http://example.org/aerospace#",
        entries=[entry],
        created_at="2025-01-01T00:00:00Z",
    )


class TestSaveLoad:
    def test_save_creates_file(self, tmp_path):
        delta = _make_delta()
        path = save(delta, tmp_path)
        assert path.exists()
        assert path.name == "delta1.json"

    def test_load_roundtrip(self, tmp_path):
        delta = _make_delta()
        save(delta, tmp_path)
        restored = load(tmp_path / "delta1.json", OntologyDelta)
        assert restored.id == delta.id
        assert restored.entries[0].triple.object == "89"

    def test_load_by_id(self, tmp_path):
        delta = _make_delta("myid")
        save(delta, tmp_path)
        restored = load_by_id("myid", tmp_path, OntologyDelta)
        assert restored.id == "myid"

    def test_list_ids(self, tmp_path):
        save(_make_delta("a1"), tmp_path)
        save(_make_delta("b2"), tmp_path)
        ids = list_ids(tmp_path)
        assert "a1" in ids
        assert "b2" in ids
        assert len(ids) == 2

    def test_exists_true(self, tmp_path):
        save(_make_delta("x"), tmp_path)
        assert exists("x", tmp_path)

    def test_exists_false(self, tmp_path):
        assert not exists("missing", tmp_path)

    def test_list_ids_empty_dir(self, tmp_path):
        assert list_ids(tmp_path) == []

    def test_list_ids_missing_dir(self, tmp_path):
        assert list_ids(tmp_path / "nonexistent") == []

    def test_sha256_str_deterministic(self):
        assert sha256_str("hello") == sha256_str("hello")

    def test_sha256_str_differs(self):
        assert sha256_str("hello") != sha256_str("world")
