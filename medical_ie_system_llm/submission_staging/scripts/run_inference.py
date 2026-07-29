"""
Entry point for Round 1 submission.

Runs the LLM-based pipeline over every file in data/raw/input/, writes
output/<doc_id>.json for each, and zips them into output.zip in the
structure the organizers require:

    output/
        1.json
        2.json
        ...
        100.json

Usage:
    python scripts/run_inference.py
    python scripts/run_inference.py --limit 5      # smoke test on 5 docs
    python scripts/run_inference.py --doc-id 1      # single doc, for debugging
"""

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from src.utils.file_io import FileLoader
from src.preprocessing.text_normalizer import DocumentNormalizer
from src.preprocessing.section_splitter import SectionSplitter
from src.extractions.llm_extractor import LlmEntityExtractor
from src.extractions.llm_medical_entity_extractor import LlmMedicalEntityExtractor
from src.llm.client import LLMClient
from src.retrieval.fuzzy_code_retriever import FuzzyCodeRetriever, format_icd10_code
from src.output.json_exporter import JsonExporter
import json

def build_pipeline(config: Config):

    llm_client = LLMClient(
        base_url=config.get("llm", "base_url"),
        model=config.get("llm", "model"),
        timeout=config.get("llm", "timeout"),
        temperature=config.get("llm", "temperature"),
        max_tokens=config.get("llm", "max_tokens"),
        keep_alive=config.get("llm", "keep_alive"),
        num_ctx=config.get("llm", "num_ctx"),
    )

    icd_retriever = FuzzyCodeRetriever(
        str(config.diagnosis_dictionary_path),
        code_col="code",
        name_col="name",
        code_formatter=format_icd10_code,
    )

    rxnorm_retriever = FuzzyCodeRetriever(
        str(config.drug_dictionary_path),
        code_col="concept_id",
        name_col="name",
    )

    llm_extractor = LlmEntityExtractor(
        llm_client,
        icd_retriever,
        rxnorm_retriever,
        candidate_top_k=config.get("llm", "candidate_top_k"),
    )

    return [
        DocumentNormalizer(),
        SectionSplitter(),
        LlmMedicalEntityExtractor(llm_extractor),
    ]

def is_already_done(doc_id: str, output_dir: Path) -> bool:
    """
    True if this doc already has valid output from a previous run — lets a
    crashed/interrupted run be resumed instead of starting over. Treats an
    empty or corrupted file (e.g. from a crash mid-write) as NOT done, so
    it gets retried rather than silently skipped.
    """
    path = output_dir / f"{doc_id}.json"
    if not path.exists():
        return False
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(content, list)
    except (json.JSONDecodeError, ValueError):
        return False


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N docs")
    parser.add_argument("--doc-id", type=str, default=None, help="Only process this one doc_id")
    args = parser.parse_args()

    config = Config()

    loader = FileLoader(config)
    documents = loader.load_all_documents()

    if args.doc_id is not None:
        documents = [d for d in documents if d.doc_id == args.doc_id]
    elif args.limit is not None:
        documents = documents[:args.limit]

    stages = build_pipeline(config)

    output_dir = Path("output")
    is_full_run = args.doc_id is None and args.limit is None

    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = JsonExporter()

    t_start = time.time()

    for i, document in enumerate(documents, start=1):

        if is_already_done(document.doc_id, output_dir):
            print(f"[{i}/{len(documents)}] doc {document.doc_id}: already done, skipping")
            continue

        t0 = time.time()

        try:
            for stage in stages:
                document = stage.process(document)

            out_path = output_dir / f"{document.doc_id}.json"
            exporter.export(document.entities, str(out_path))

            elapsed = time.time() - t0
            print(f"[{i}/{len(documents)}] doc {document.doc_id}: "
                  f"{len(document.entities)} entities in {elapsed:.1f}s")

        except Exception as e:
            # A single bad document (network hiccup, malformed response,
            # anything unexpected) must not kill a ~7-hour unattended run.
            # No output file gets written for this doc, so a later re-run
            # will correctly retry it via is_already_done() above.
            print(f"[{i}/{len(documents)}] doc {document.doc_id}: FAILED — {type(e).__name__}: {e}")
            print(f"  (no output written for this doc — re-run the script later to retry it)")

    total_elapsed = time.time() - t_start
    print(f"\nDone: {len(documents)} docs in {total_elapsed:.1f}s "
          f"({total_elapsed / max(len(documents), 1):.1f}s/doc avg)")

    if is_full_run:
        zip_path = Path("output.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for json_file in sorted(output_dir.glob("*.json")):
                zf.write(json_file, arcname=f"output/{json_file.name}")
        print(f"Wrote {zip_path}")
    else:
        print("(test run — skipping output.zip; run with no --doc-id/--limit for the real submission zip)")

if __name__ == "__main__":
    main()
