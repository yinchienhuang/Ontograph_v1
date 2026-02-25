"""
utils/io.py — Content-addressed JSON persistence for pipeline artifacts.

Every artifact is saved as `<data_dir>/<stage>/<id>.json`.
Loading is type-safe: pass the Pydantic model class and get back a
validated instance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Core save / load
# ---------------------------------------------------------------------------

def save(artifact: BaseModel, directory: str | Path) -> Path:
    """
    Serialize `artifact` to JSON and write it to `<directory>/<id>.json`.

    The file name is taken from `artifact.id` if the model has an `id` field;
    otherwise falls back to sha256 of the content.

    Returns the path written.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = artifact.model_dump_json(indent=2)

    artifact_id: str = getattr(artifact, "id", None) or _sha256_str(payload)
    path = out_dir / f"{artifact_id}.json"
    path.write_text(payload, encoding="utf-8")
    return path


def load(path: str | Path, model: type[T]) -> T:
    """
    Read a JSON file and parse it into `model`.

    Raises FileNotFoundError if the file doesn't exist.
    Raises pydantic.ValidationError if the JSON doesn't match the schema.
    """
    text = Path(path).read_text(encoding="utf-8")
    return model.model_validate_json(text)


def load_by_id(artifact_id: str, directory: str | Path, model: type[T]) -> T:
    """Convenience wrapper: load `<directory>/<artifact_id>.json`."""
    return load(Path(directory) / f"{artifact_id}.json", model)


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------

def list_ids(directory: str | Path) -> list[str]:
    """Return the stem (id) of every *.json file in `directory`."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def exists(artifact_id: str, directory: str | Path) -> bool:
    return (Path(directory) / f"{artifact_id}.json").exists()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def sha256_str(text: str) -> str:
    """Return the hex SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode()).hexdigest()
