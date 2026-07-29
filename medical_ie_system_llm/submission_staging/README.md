# Viettel AI Race 2026 — Đề 2: Ontological Reasoning in Medical Knowledge Retrieval

*Team name:* VLUS 

*Members:* 1. Nguyễn Minh Trung

## 1. Overview

This system extracts medical concepts (symptoms, lab tests, lab results,
diagnoses, drugs) from free-form Vietnamese clinical text, maps diagnoses to
ICD-10 and drugs to RxNorm, and infers contextual assertions (negation,
family history, past history) — per the Round 1 problem statement.

*Approach:* a self-hosted, locally-run LLM (Qwen3-4B via Ollama, ≤9B
params as required by the rules, no external API calls) performs the
extraction in one call per document, few-shot prompted on the organizers'
own examples. Diagnosis/drug candidates are grounded via fuzzy retrieval
against the real ICD-10/RxNorm corpora rather than trusted from the model's
own memory of exact codes, since LLMs are unreliable sources of precise
code numbers.

## 2. Setup

### Requirements
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- ~6GB free disk space for the model weights

### Install
pip install -r requirements.txt --break-system-packages
ollama pull qwen3:4b
ollama serve

### Model
This submission uses *Qwen3-4B*, pulled via Ollama's public model
registry (ollama pull qwen3:4b), run entirely locally — no external API
calls are made at any point during inference, satisfying the competition's
self-host requirement. We do not ship the raw model weight files in this
submission, since they are a standard, publicly available checkpoint
reproducible via the single command above; the exact model tag pinned is
qwen3:4b as resolved by Ollama at the time of submission.

## 3. Running inference

python scripts/run_inference.py

This processes every file in data/raw/input/, writes output/<doc_id>.json
for each, and packages them into output.zip in the required submission
structure:
output/
    1.json
    2.json
    ...
    100.json

*Resumable:* if interrupted (crash, closed terminal, network drop), simply
re-run the same command — completed documents are detected and skipped, so
no work is lost or duplicated.

Useful flags for testing/debugging a subset instead of the full corpus:
python scripts/run_inference.py --doc-id 1     # single document
python scripts/run_inference.py --limit 5      # first N documents

### Validation (optional, no LLM required)
python scripts/validate_output.py
Runs a set of offline heuristic checks (overlapping spans, suspicious span
lengths, position/text consistency, etc.) against whatever is currently in
output/, and writes validation_report.md. Safe to run at any time,
including while inference is still in progress on other documents.

## 4. Architecture

raw .txt file
     │
     ▼
DocumentNormalizer      — whitespace cleanup; builds an offset map back to
     │                     the original file so exported positions are
     │                     always correct against the raw input, not a
     │                     normalized copy
     ▼
SectionSplitter         — splits into numbered sections where present;
     │                     falls back to treating the whole document as one
     │                     section otherwise (handles headerless documents)
     ▼
LlmMedicalEntityExtractor
     │
     ▼
LlmEntityExtractor      — one LLM call per document:
     │                     1. build prompt (few-shot examples + full doc)
     │                     2. call the local LLM
     │                     3. parse + validate the JSON response
     │                     4. locate each claimed entity span in the source
     │                        text; DROP any entity whose text can't
     │                        actually be found (hallucination guard)
     │                     5. for CHẨN_ĐOÁN/THUỐC, retrieve real ICD-10 /
     │                        RxNorm candidates via fuzzy search — never
     │                        trust a code the model states from memory
     │                     6. merge exact-duplicate entities (same text,
     │                        type, and position emitted more than once)
     ▼
JsonExporter            — writes the final per-document JSON in the exact
                           schema specified by the competition

### Key design decisions
- *Retrieval over generation for codes.* The LLM proposes a canonical
  English medical term (lookup_term); a separate fuzzy-matching step
  (rapidfuzz) searches the real ICD-10/RxNorm corpora for the closest
  official codes. This grounds every candidate in an actual entry from the
  official code lists.
- *Hallucination guard.* Every entity's text field is verified to
  actually appear (verbatim, whitespace-tolerant) in the source document
  before being accepted. If not found, it's dropped rather than exported
  with a fabricated position.
- *One LLM call per document, not per section.* Reduces fixed prompt
  overhead that would otherwise be paid multiple times per document.
- *Prompt engineering for genre coverage.* The Round 1 dataset contains a
  mix of formal clinical notes, patient/doctor Q&A forum posts, and
  educational articles. The prompt explicitly handles each: in Q&A text,
  the first-person patient is treated as the subject for assertions;
  generic/educational content is extracted but not assigned
  patient-specific assertions; redacted drug names (shown as ***...) are
  still extracted with lookup_term: null.

## 5. Known limitations

- Rare/compound diagnosis or drug phrasings occasionally retrieve no or
  imprecise ICD-10/RxNorm candidates, since fuzzy string matching (not
  semantic search) is used for retrieval.
- Occasional over-broad spans on long, narrative sentences.
- A small number of non-clinical mentions (risk factors, procedures) are
  sometimes mistyped rather than omitted.
- Two documents in the Round 1 test set (near-duplicate Q&A posts with
  repetitive phrasing) consistently and deterministically triggered the
  model to echo content from one of the prompt's few-shot examples instead
  of extracting from the actual input. This was reproduced across multiple
  runs, including with increased sampling temperature. Every fabricated
  entity was correctly caught and dropped by the hallucination guard (see
  SpanLocator), so these two documents export as empty result sets rather
  than incorrect ones — a bounded, contained failure rather than a silent
  data-quality issue.

These are documented tradeoffs made under the competition's time and
compute constraints (self-hosted model, ≤9B parameters, no external API).

## 6. Repository structure

app/                    configuration loading
src/
  models/                data classes (Document, Entity, Section, ...)
  preprocessing/          text normalization, section splitting, span location
  llm/                    LLM client, prompts, response parsing
  extractions/            main extraction orchestration
  retrieval/              ICD-10 / RxNorm fuzzy candidate retrieval
  output/                 JSON export
data/
  knowledge/              icd10.csv, drugs.csv (processed code corpora)
  raw/input/              competition input documents
scripts/
  run_inference.py        main entry point
  validate_output.py      offline QA validation
configs/config.yml        pipeline + model configuration
tests/                    regression tests