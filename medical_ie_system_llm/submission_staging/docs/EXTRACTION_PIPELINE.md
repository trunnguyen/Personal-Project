# LLM Extraction Pipeline — How It Works

This documents the code built on **July 20** — the LLM-based extraction
module that replaces the four rule-based extractors. Read this before
tomorrow's session; each new day's code will get a section added here.

## 1. The big picture

```
raw .txt file
     │
     ▼
DocumentNormalizer    → cleans whitespace, builds offset_map (normalized → original position)
     │
     ▼
SectionSplitter        → splits into numbered sections (with fallbacks for headerless docs)
     │
     ▼
LlmMedicalEntityExtractor   → loops sections, calls LlmEntityExtractor per section
     │
     ▼
LlmEntityExtractor      → prompt → LLM → parse → locate span → map offset → retrieve candidates
     │
     ▼
JsonExporter            → writes <doc_id>.json in the exact competition schema
```

One LLM call happens per **section**, not per document — this keeps each
call focused on a manageable chunk of text and matches how the rule-based
version was scoped, so section-based assertions (like `isHistorical` from
section title) still make sense contextually to the model.

## 2. File-by-file

### `src/llm/client.py`
`LLMClient` — sends a chat request to whatever OpenAI-compatible endpoint
you point it at (`base_url` in `configs/config.yml`, defaults to Ollama's
`http://localhost:11434/v1`). Disables Qwen3's "thinking mode" explicitly,
since we want direct JSON, not a reasoning trace, for a batch job.

`FakeLLMClient` — a drop-in replacement used only in tests. You give it a
dict of `{prompt_substring: canned_response}` and it returns the canned
response instead of calling a real model. This is how I tested the whole
pipeline today without ever running Qwen3-8B.

### `src/llm/prompts.py`
The system prompt (task definition in Vietnamese, matching the organizers'
own terminology) plus three few-shot examples:
1. The organizers' official GERD example (diagnosis + symptoms + labs)
2. The organizers' official drug-list example (dosed drugs + `isHistorical`)
3. A negation example I added, since neither official example demonstrates
   `isNegated` and that was the single biggest gap in the old rule-based
   system

Each example teaches the model to also output a `lookup_term` field for
`CHẨN_ĐOÁN`/`THUỐC` — an English canonical name used later for retrieval
(see §4). The model is *not* asked to produce ICD-10/RxNorm codes directly
— it doesn't reliably know exact code numbers, so codes always come from
real lookup, never from the model's memory.

### `src/llm/response_parser.py`
Defensive JSON parsing: strips markdown code fences if the model adds them
despite instructions, validates every field, drops individual malformed
entities instead of failing the whole section, and strips
`assertions`/`lookup_term` from entity types that shouldn't have them
(defense in depth — even if the model ignores the schema instructions, bad
fields never reach the output).

### `src/preprocessing/span_locator.py`
**The hallucination guard.** The model returns `"text"` as a claimed
verbatim quote. `SpanLocator.locate()` actually searches for it in the real
section text (whitespace/case-tolerant, but content must match exactly). If
it's not found — model paraphrased, invented, or mistranscribed it — the
entity is dropped. No entity ever reaches the output with a fabricated
position.

Also handles repeated mentions: if "táo bón" appears twice in the source
tied to two different drugs (as in the official example), it resolves each
occurrence separately, left to right, rather than collapsing both onto the
first match.

### `src/retrieval/fuzzy_code_retriever.py`
Turns `icd10.csv`/`drugs.csv` from exact-match dictionaries into a
**retrieval corpus**. Given the model's English `lookup_term`, fuzzy-searches
(via `rapidfuzz`) the real code list and returns the top-K matching codes.
This is what grounds `candidates` in real, valid codes.

`format_icd10_code()` also hedges against a version mismatch I found: the
source ICD-10 file uses newer, more subdivided codes than the organizers'
own example does (`K21.00`/`K21.01` vs. their `K21.0`), so it emits both the
precise code and its classic parent as separate candidates.

### `src/extractions/llm_extractor.py` / `llm_medical_entity_extractor.py`
Orchestration. The extractor builds the prompt, calls the model, parses the
response, locates each span, converts the position through `offset_map` (so
the final position is relative to your *original* raw `.txt` file, not the
whitespace-normalized text — this was a real, previously-unused bug fix),
and runs candidate retrieval for diagnosis/drug entities.

### `src/output/json_exporter.py`
Now conditionally omits `assertions`/`candidates` keys for entity types
that shouldn't have them (matches the organizers' example schema exactly,
rather than always emitting an empty list).

### `scripts/run_inference.py`
The actual submission entry point. Run this to produce `output.zip`.

## 3. How to test it

**Without a running model** (already done today, re-runnable anytime):
```bash
python3 tests/llm_pipeline/test_e2e_offset_mapping.py   # full pipeline against real doc 1, scripted LLM response
python3 tests/llm_pipeline/test_guards.py                # hallucination guard, whitespace tolerance, malformed JSON
```
These prove the scaffolding is correct. They do **not** prove extraction
quality — that requires a real model.

**With a running model:**
```bash
# 1. Start the model
ollama serve
ollama pull qwen3:8b   # first time only

# 2. Smoke test on one document
python scripts/run_inference.py --doc-id 1

# 3. Once that looks right, a small batch
python scripts/run_inference.py --limit 5

# 4. Full run (produces output.zip)
python scripts/run_inference.py
```
Each run prints per-document entity counts and timing, and a total average
— watch that average once running for real, since LLM inference is much
slower than the old rule-based pass and we need to know if the full 100-doc
run takes minutes or hours.

## 4. Design decisions worth knowing about

- **Why retrieval instead of asking the LLM for codes directly?** LLMs
  hallucinate exact code numbers. Retrieval grounds every candidate in a
  real row from the official ICD-10/RxNorm files.
- **Why English `lookup_term` instead of matching Vietnamese directly?**
  The ICD-10/RxNorm source data is English. Rather than hand-translating
  ~75k codes, the model (which has genuine bilingual medical knowledge)
  proposes the English term, and fuzzy retrieval does the rest.
- **Why per-section LLM calls instead of one call per document?** Keeps
  each call focused, keeps prompts shorter (cheaper, faster, less drift),
  and section-level context (like the title driving `isHistorical`) stays
  meaningful to the model.
