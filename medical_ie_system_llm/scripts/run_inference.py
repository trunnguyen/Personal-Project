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


def build_pipeline(config: Config):

    llm_client = LLMClient(
        base_url=config.get("llm", "base_url"),
        model=config.get("llm", "model"),
        timeout=config.get("llm", "timeout"),
        temperature=config.get("llm", "temperature"),
        max_tokens=config.get("llm", "max_tokens"),
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
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    exporter = JsonExporter()

    t_start = time.time()

    for i, document in enumerate(documents, start=1):

        t0 = time.time()

        for stage in stages:
            document = stage.process(document)

        out_path = output_dir / f"{document.doc_id}.json"
        exporter.export(document.entities, str(out_path))

        elapsed = time.time() - t0
        print(f"[{i}/{len(documents)}] doc {document.doc_id}: "
              f"{len(document.entities)} entities in {elapsed:.1f}s")

    total_elapsed = time.time() - t_start
    print(f"\nDone: {len(documents)} docs in {total_elapsed:.1f}s "
          f"({total_elapsed / max(len(documents), 1):.1f}s/doc avg)")

    zip_path = Path("output.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for json_file in sorted(output_dir.glob("*.json")):
            zf.write(json_file, arcname=f"output/{json_file.name}")

    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
