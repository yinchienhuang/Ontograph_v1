"""
converters/text.py — Convert a plain-text file to a normalized RawDocument.

Heading detection heuristics (applied per line):
  - Numbered sections: "1 Introduction", "1.2 Background", "1.2.3 Details"
    → level = number of dotted components (capped at 3)
  - Short all-caps lines without terminal punctuation: "INTRODUCTION"
    → level 2
  - Everything else passes through as-is.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from ontograph.models.document import RawDocument

# e.g. "1 Intro", "1.2 Background", "3.4.1 Detail"  (starts with digit[.digit]* space)
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\s+[A-Z\w]")


def _heading_level(line: str) -> int | None:
    """Return heading level 1–3 for heading-like lines, None otherwise."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    m = _NUMBERED.match(stripped)
    if m:
        depth = m.group(1).count(".") + 1
        return min(depth, 3)

    # Short all-caps line (at least 3 chars, no terminal sentence punctuation)
    if (
        stripped == stripped.upper()
        and len(stripped) >= 3
        and stripped[-1] not in ".,:;?!"
        and stripped.replace(" ", "").isalpha()
    ):
        return 2

    return None


def convert_text(path: str | Path) -> RawDocument:
    """Convert a plain-text file to a RawDocument."""
    path = Path(path).resolve()
    raw_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    doc_id = hashlib.sha256(
        f"{path}:{source_sha256}".encode()
    ).hexdigest()[:16]

    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    md_lines: list[str] = []
    for line in lines:
        level = _heading_level(line)
        if level is not None:
            md_lines.append("#" * level + " " + line.strip())
        else:
            md_lines.append(line)

    markdown = "\n".join(md_lines)

    return RawDocument(
        id=doc_id,
        source_path=str(path),
        source_format="txt",
        source_sha256=source_sha256,
        markdown=markdown,
        page_map=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
