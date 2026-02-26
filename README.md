# Ontograph v1

A research pipeline that ingests engineering documents into an OWL ontology and evaluates
whether **structured knowledge representations** improve LLM-based reasoning compared to
prose documentation alone.

Three research comparisons are built in:

| Module | Question |
|--------|----------|
| **Rule Checker** (`check_rules --mode both`) | Does exact OWL data help an LLM detect rule violations better than vague documentation prose? |
| **Impact Analysis** (`analyze_impact --mode both`) | After a design change, does the updated ontology allow an LLM to correctly predict the new violation state vs. stale documents? |
| **OWL Evaluator** (`run_pipeline --from-owl`) | How accurately does the pipeline reconstruct an existing ontology from a synthesized document (Recall / Precision / F1)? |

---

## Architecture

```
Document (PDF / TXT / MD)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Module 1 — Ingest + Ontology                            │
│  Convert → Chunk → Extract (LLM) → Map → Align → Review │
│                          ↓                               │
│                    Working OWL                           │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Module 2 — Document Synthesizer                         │
│  Working OWL → LLM → Grounded Markdown + Self-Check     │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Module 3 — Evaluation / Impact Analysis                 │
│  Rule Checker · OWL Evaluator · Impact Analyzer          │
└──────────────────────────────────────────────────────────┘
```

All LLM calls use structured Pydantic outputs — no free-text parsing. Three providers
are supported: **Claude** (default), **GPT-4o**, **Gemini**.

---

## Installation

**Requires Python 3.11+**

```bash
git clone https://github.com/yinchienhuang/Ontograph_v1.git
cd Ontograph_v1
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e .
```

Copy `.env.example` to `.env` and add at least one API key:

```
ANTHROPIC_API_KEY=sk-ant-...   # Claude (default)
OPENAI_API_KEY=sk-...          # GPT-4o
GOOGLE_API_KEY=...             # Gemini
```

---

## Quick Start

The repository ships with a pre-built CubeSat example in `data/`. You can run all
three research comparisons immediately without ingesting any new documents.

### 1 — Check rule violations (structured OWL vs. prose document)

```bash
.venv/Scripts/python scripts/check_rules.py \
    --rules    data/rules_example.yaml \
    --owl      data/ontology/working.owl \
    --doc      data/raw/cubesatontology_synthesized.md \
    --provider claude \
    --mode     both
```

### 2 — Impact analysis (design change propagation)

```bash
.venv/Scripts/python scripts/analyze_impact.py \
    --scenarios data/impact_scenarios_example.yaml \
    --rules     data/rules_example.yaml \
    --owl       data/ontology/working.owl \
    --doc       data/raw/cubesatontology_synthesized.md \
    --provider  claude \
    --mode      both
```

### 3 — Full validation loop (ingest synthesized doc → compare against source OWL)

```bash
.venv/Scripts/python scripts/run_pipeline.py \
    --from-owl data/ontology/cubesatontology.owl \
    --provider claude \
    --auto-approve
```

---

## Pipeline Steps

The main orchestrator is `scripts/run_pipeline.py`.

| Step | Name | Description |
|------|------|-------------|
| 0 | **Synthesize** | OWL → LLM → Markdown (validation-loop mode only) |
| 1 | **Convert + Chunk** | PDF/TXT/MD → hierarchy-aware chunks |
| 2 | **Extract + Map** | LLM entity extraction → proposed OWL triples |
| 3 | **Align** | Embedding + LLM deduplication of entities |
| 4 | **Review** | Human approval / rejection / edit of each triple |
| 5 | **Write OWL** | Approved triples appended to `working.owl` |
| 6 | **Compare** | Source vs. working OWL: Recall / Precision / F1 |
| 7 | **Rule Check** | LLM rule violation check (ontology / document / both) |
| 8 | **Impact Analysis** | Design-change impact: pre- vs. post-change rule state |

```bash
# Validation loop with all optional steps
.venv/Scripts/python scripts/run_pipeline.py \
    --from-owl    data/ontology/cubesatontology.owl \
    --provider    claude \
    --auto-approve \
    --rules       data/rules_example.yaml \
    --rules-mode  both \
    --impact      data/impact_scenarios_example.yaml \
    --impact-mode both
```

---

## Standalone Scripts

| Script | Description |
|--------|-------------|
| `scripts/check_rules.py` | Rule violation checker — ontology / document / both modes |
| `scripts/analyze_impact.py` | Design-change impact analysis — scores each arm against ground truth |
| `scripts/review_delta.py` | Interactive human review of proposed OWL triples |
| `scripts/align_delta.py` | Deduplicate entities in a delta file |
| `scripts/apply_org_knowledge.py` | Inject organizational knowledge triples from YAML |
| `scripts/demo_synthesizer.py` | Quick synthesizer demo from hard-coded data |

---

## Key Data Files

| File | Description |
|------|-------------|
| `data/ontology/cubesatontology.owl` | TBox — class hierarchy and property definitions |
| `data/ontology/working.owl` | Working OWL — TBox + approved ABox individuals |
| `data/rules_example.yaml` | Four CubeSat compatibility rules (all violated in current OWL) |
| `data/impact_scenarios_example.yaml` | Four design-change scenarios (each resolves one rule) |
| `data/raw/cubesatontology_synthesized.md` | LLM-synthesized document from the working OWL |
| `data/violations/` | Saved `ViolationReport` JSON files |
| `data/evaluations/` | Saved `ImpactAnalysisResult` JSON files |

---

## Module Overview

```
ontograph/
├── ingest/         Document loading, chunking, LLM extraction, OWL mapping, alignment
├── llm/            Unified provider abstraction (Claude, GPT-4o, Gemini)
├── models/         Pydantic v2 data contracts for all pipeline stages
├── synthesizer/    OWL → grounded Markdown with provenance + self-check
├── evaluator/      Source vs. working OWL: Recall / Precision / F1
├── rules/          Structured YAML rules → LLM violation checker (ontology + document modes)
├── impact/         Design-change impact analysis wrapping the rule checker
└── utils/          Artifact I/O (content-addressed JSON), OWL / rdflib helpers
```

---

## Rules and Impact Scenarios

### Rules (`data/rules_example.yaml`)

Rules are YAML threshold conditions paired with a vague plain-English description (generated
by the LLM to simulate incomplete documentation):

```yaml
namespace: "http://example.org/cubesat-ontology#"
rules:
  - id: "compat-002"
    name: "Radio Power Draw Heatsink Requirement"
    subject_type: "CommunicationSubsystem"
    object_type:  "Radio"
    when: { attribute: "powerW", operator: ">", value: 3.0, unit: "W" }
    severity: "warning"
```

Run `check_rules.py --mode both` to compare how well each arm detects violations.

### Impact Scenarios (`data/impact_scenarios_example.yaml`)

Each scenario specifies a design change and the expected post-change violation state
(ground truth from an external OWL reasoner):

```yaml
scenarios:
  - id: "impact-002"
    description: "Upgrade Radio1 to a lower-power model (4.0W → 2.0W)"
    component_local: "Radio1"
    attribute_changes:
      - property_local: "powerW"
        old_value: "4.0"
        new_value: "2.0"
    ground_truth_violations:     # what should fire AFTER the change
      - "compat-001"
      - "compat-003"
      - "single-001"
```

---

## Running Tests

```bash
.venv/Scripts/python -m pytest -q      # 451 tests, ~5 s
```

---

## Documentation

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — Full step-by-step pipeline guide
- [`docs/rules_checker.md`](docs/rules_checker.md) — Rule checker reference

---

## Project Structure

```
Ontograph_v1/
├── ontograph/              Main package
├── scripts/                Standalone CLI scripts
├── tests/unit/             Unit tests (451 passing)
├── data/
│   ├── ontology/           OWL files (TBox + working)
│   ├── raw/                Source documents + synthesized Markdown
│   ├── rules_example.yaml
│   ├── impact_scenarios_example.yaml
│   ├── violations/         Rule check results (JSON)
│   └── evaluations/        Impact analysis results (JSON)
├── docs/                   Guides
└── pyproject.toml
```
