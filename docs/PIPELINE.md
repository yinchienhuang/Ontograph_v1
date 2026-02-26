# Ontograph v1 — Pipeline Run Guide

This document describes how to run every script in the pipeline end-to-end,
from a raw document to a populated, reviewed OWL ontology.

---

## Prerequisites

### 1. Python environment

```bash
# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Or invoke scripts directly with:
.venv/Scripts/python scripts/<script>.py ...
```

### 2. API keys

Copy `.env.example` to `.env` (if it exists) and fill in at least one key:

```
ANTHROPIC_API_KEY=sk-ant-...   # Claude (default provider)
OPENAI_API_KEY=sk-...          # GPT-4o
GOOGLE_API_KEY=...             # Gemini
```

> **Phase 1 (chunking only) requires no API key.**
> Phases 2–4 require a key for whichever `--provider` you choose.

### 3. Input files

| File | Description |
|------|-------------|
| `data/raw/<doc>.pdf` | Source document (PDF, TXT, or MD) |
| `data/ontology/cubesatontology.owl` | TBox schema (classes + properties) |
| `data/org_knowledge_example.yaml` | Template for organizational knowledge rules |

---

## Step 1 — Ingest a Document

### Phase 1: Convert + Chunk (no LLM, instant)

Converts a document to Markdown and splits it into chunks aligned with the
section hierarchy.  No API key required.

```bash
.venv/Scripts/python scripts/test_ingest.py data/raw/cds_rev13_final2.pdf
```

**Output files:**

| Path | Contents |
|------|----------|
| `data/raw/<stem>.md` | Converted Markdown (for inspection) |
| `data/artifacts/<id>.json` | `DocumentArtifact` — chunk list with section paths |

---

### Phase 2: Extract + Map (requires API key)

Runs LLM entity extraction (one call per chunk) and OWL mapping (one call per
entity).  With `--tbox` the mapper is given the ontology vocabulary and maps
to existing class/property names instead of inventing new ones.

```bash
# Full document, TBox-aware mapping
.venv/Scripts/python scripts/test_ingest.py data/raw/cds_rev13_final2.pdf \
    --extract \
    --tbox data/ontology/cubesatontology.owl \
    --provider claude

# Cheaper test — first 5 chunks only
.venv/Scripts/python scripts/test_ingest.py data/raw/cds_rev13_final2.pdf \
    --extract \
    --tbox data/ontology/cubesatontology.owl \
    --max-chunks 5

# Provider choices: claude (default), gpt-4o, gemini
```

**Output files:**

| Path | Contents |
|------|----------|
| `data/extractions/<id>.json` | `ExtractionBundle` — extracted entities + attributes |
| `data/deltas/<id>.json` | `OntologyDelta` — proposed triples (status = "proposed") |

Each proposed triple includes:
- `rdf:type` class assertion
- `rdfs:comment` description of the individual (from LLM)
- One triple per extracted attribute (e.g., `massKg`, `powerW`)

---

## Step 2 — Review the Delta

Interactively review every proposed triple.  You approve, reject, or edit
each one.  The delta JSON is updated in place after the session.

```bash
# Review only (no OWL write)
.venv/Scripts/python scripts/review_delta.py data/deltas/<id>.json

# Review + write approved triples to working OWL
# --tbox seeds the output file with the full class hierarchy first
.venv/Scripts/python scripts/review_delta.py data/deltas/<id>.json \
    --owl  data/ontology/working.owl \
    --tbox data/ontology/cubesatontology.owl
```

**Review commands:**

| Key | Action |
|-----|--------|
| `a` | Approve triple as-is |
| `r` | Reject triple (recorded in delta, not written to OWL) |
| `e` | Edit the object value inline, then approve |
| `s` | Skip — leave status as "proposed" for later |
| `q` | Quit (all changes made so far are saved) |

**Output:**

- `data/deltas/<id>.json` — updated with per-entry status
- `data/ontology/working.owl` (if `--owl` provided) — contains:
  - Full TBox class hierarchy from `cubesatontology.owl` (no instances from TBox)
  - Only the approved ABox triples from this delta

> **Re-running is safe.** `copy_tbox` is idempotent — re-merging the same
> TBox into an existing working OWL will not duplicate triples.

---

## Step 3 — Add Organizational Knowledge (optional)

Encode tacit engineering rules that don't appear in documents — compatibility
constraints, safety requirements, supersession rules, etc.

### 3a. Author the YAML

Copy the template and edit it:

```bash
cp data/org_knowledge_example.yaml data/org_knowledge.yaml
# Edit data/org_knowledge.yaml in any text editor
```

Each rule follows this structure:

```yaml
rules:
  - subject:    "http://example.org/cubesat-ontology#BatteryPack_A"
    predicate:  "http://example.org/cubesat-ontology#massKg"
    object:     "1.2"
    datatype:   "http://www.w3.org/2001/XMLSchema#decimal"
    note:       "Confirmed by systems engineering team — PDR review 2024-11"
```

### 3b. Preview rules (dry-run)

```bash
.venv/Scripts/python scripts/apply_org_knowledge.py \
    --yaml data/org_knowledge.yaml \
    --owl  data/ontology/working.owl \
    --dry-run
```

### 3c. Apply to OWL

```bash
.venv/Scripts/python scripts/apply_org_knowledge.py \
    --yaml data/org_knowledge.yaml \
    --owl  data/ontology/working.owl
```

---

## Step 4 — Run the Synthesizer Demo (optional)

Generates a grounded Markdown report from a small hard-coded delta.
Useful for testing the synthesizer without running the full pipeline.

```bash
.venv/Scripts/python scripts/demo_synthesizer.py                    # Claude
.venv/Scripts/python scripts/demo_synthesizer.py --provider gpt-4o  # GPT-4o
.venv/Scripts/python scripts/demo_synthesizer.py --save             # save .md output
```

---

## Data Directory Layout

```
data/
├── raw/
│   ├── cds_rev13_final2.pdf     ← source document
│   └── cds_rev13_final2.md      ← converted Markdown (generated)
├── artifacts/
│   └── <sha>.json               ← DocumentArtifact (chunks)
├── extractions/
│   └── <sha>.json               ← ExtractionBundle (entities)
├── deltas/
│   └── <sha>.json               ← OntologyDelta (proposed → approved)
├── ontology/
│   ├── cubesatontology.owl      ← TBox (source of truth, never modified)
│   └── working.owl              ← output: TBox + approved ABox triples
└── org_knowledge_example.yaml   ← template for organizational rules
```

---

## Typical End-to-End Run

```bash
# 1. Ingest and extract (first 10 chunks for a quick test)
.venv/Scripts/python scripts/test_ingest.py data/raw/cds_rev13_final2.pdf \
    --extract --max-chunks 10 \
    --tbox data/ontology/cubesatontology.owl

# 2. Review — note the delta ID printed by step 1
.venv/Scripts/python scripts/review_delta.py data/deltas/<delta-id>.json \
    --owl  data/ontology/working.owl \
    --tbox data/ontology/cubesatontology.owl

# 3. (Optional) Add org knowledge
.venv/Scripts/python scripts/apply_org_knowledge.py \
    --yaml data/org_knowledge.yaml \
    --owl  data/ontology/working.owl
```

After step 2, `data/ontology/working.owl` contains the full CubeSat class
hierarchy plus every approved instance triple — ready to load in Protégé or
feed into the evaluator.

---

## Running Tests

```bash
# All tests (263 tests, ~5 s)
.venv/Scripts/python -m pytest tests/ -q

# Single module
.venv/Scripts/python -m pytest tests/unit/test_ingest_mapper.py -v
```
