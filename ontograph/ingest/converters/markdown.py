"""
converters/markdown.py — Load an existing Markdown file as a RawDocument.

No heading transformation is applied — the file is already normalized
Markdown with `#` syntax. Three or more consecutive blank lines are
collapsed to two to keep the output clean for the chunker.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from ontograph.models.document import RawDocument

_EXCESS_BLANK = re.compile(r"\n{3,}")


def convert_markdown(path: str | Path) -> RawDocument:
    """Load a Markdown file as a RawDocument (no structural transformation)."""
    path = Path(path).resolve()
    raw_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    doc_id = hashlib.sha256(
        f"{path}:{source_sha256}".encode()
    ).hexdigest()[:16]

    text = raw_bytes.decode("utf-8", errors="replace")
    # Collapse 3+ consecutive blank lines → 2
    markdown = _EXCESS_BLANK.sub("\n\n", text).strip() + "\n"

    return RawDocument(
        id=doc_id,
        source_path=str(path),
        source_format="md",
        source_sha256=source_sha256,
        markdown=markdown,
        page_map=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
