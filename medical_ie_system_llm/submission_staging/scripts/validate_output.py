"""
QA validation pass over output/*.json files. Doesn't call the LLM — pure
heuristic checks against the raw input text, so it runs instantly and can
be re-run after every inference run without cost.

This does NOT tell you an entity is definitely wrong — it flags entities
that match known failure patterns we've actually observed in testing, so
you can spot-check a short, prioritized list instead of reading 100 files
end to end.

Usage:
    python scripts/validate_output.py
    python scripts/validate_output.py --output-dir output --input-dir data/raw/input
    python scripts/validate_output.py --doc-id 39      # single doc, for debugging
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Per-type span-length thresholds (characters). Calibrated from real
# examples we've seen: TRIỆU_CHỨNG/TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM can
# legitimately be longish narrative phrases ("chụp x-quang ngực có xẹp phổi
# thùy dưới phải do chèn ép kèm tràn dịch màng phổi" is real and correct at
# 79 chars); THUỐC and CHẨN_ĐOÁN are usually much shorter, so a long span
# there is a stronger signal of sentence-swallowing.
LONG_SPAN_THRESHOLDS = {
    "THUỐC": 60,
    "CHẨN_ĐOÁN": 60,
    "TRIỆU_CHỨNG": 90,
    "TÊN_XÉT_NGHIỆM": 90,
    "KẾT_QUẢ_XÉT_NGHIỆM": 90,
}

PROCEDURE_KEYWORDS = [
    "phẫu thuật", "sinh thiết", "chọc dò", "nội soi", "cắt bỏ",
    "cắt nối", "mổ ", "thăm dò", "đặt catheter", "đặt stent",
]

RISK_FACTOR_KEYWORDS = [
    "căng thẳng", "mất việc", "cà phê", "hút thuốc", "hôn nhân",
    "thất nghiệp", "áp lực công việc",
]

NEGATION_CUES = ["không", "chưa", "phủ nhận", "loại trừ"]
FAMILY_CUES = ["mẹ", "bố", "cha", "anh", "chị",
               "ông", "bà", "gia đình"]

# Same clause-boundary logic as src/assertions/assertions.py (kept in sync
# deliberately): newline, sentence-ending punctuation, semicolon, or a
# dash-bullet at line start end a clause. Commas do NOT end a clause, since
# Vietnamese negation scopes over comma-separated lists (e.g. "Không buồn
# nôn, hay nôn, đổ mồ hôi" — the negation covers the whole list).
CLAUSE_BOUNDARY = re.compile(r"[\n;.]|(?<=\n)\s*-")

# How far back (characters) to look for a cue, but bounded by the clause
# boundary above — whichever is closer.
CONTEXT_WINDOW = 60


def load_entities(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_position_text_mismatch(entities, raw_text, issues):
    for e in entities:
        start, end = e["position"]
        actual = raw_text[start:end]
        if actual != e["text"]:
            issues.append({
                "check": "position_text_mismatch",
                "severity": "critical",
                "entity_text": e["text"],
                "detail": f"raw_text[{start}:{end}] = {actual!r}, but entity text = {e['text']!r}",
            })


def check_long_span(entities, issues):
    for e in entities:
        threshold = LONG_SPAN_THRESHOLDS.get(e["type"], 90)
        if len(e["text"]) > threshold:
            issues.append({
                "check": "suspiciously_long_span",
                "severity": "warning",
                "entity_text": e["text"],
                "detail": f"{len(e['text'])} chars (threshold {threshold} for {e['type']}) — possible sentence-swallowing",
            })


def check_overlapping_spans(entities, issues):
    sorted_entities = sorted(entities, key=lambda e: e["position"][0])
    for i in range(len(sorted_entities) - 1):
        a, b = sorted_entities[i], sorted_entities[i + 1]
        a_start, a_end = a["position"]
        b_start, b_end = b["position"]
        if b_start < a_end:  # overlap
            issues.append({
                "check": "overlapping_spans",
                "severity": "warning",
                "entity_text": f"{a['text']!r} <-> {b['text']!r}",
                "detail": f"positions {a['position']} and {b['position']} overlap",
            })


def check_exact_duplicates(entities, issues):
    seen = {}
    for e in entities:
        key = (e["text"], e["type"], tuple(e["position"]))
        if key in seen:
            issues.append({
                "check": "exact_duplicate",
                "severity": "warning",
                "entity_text": e["text"],
                "detail": f"duplicate entity at {e['position']}",
            })
        seen[key] = True


def check_procedure_as_drug(entities, issues):
    for e in entities:
        if e["type"] != "THUỐC":
            continue
        text_lower = e["text"].lower()
        for kw in PROCEDURE_KEYWORDS:
            if kw in text_lower:
                issues.append({
                    "check": "suspected_procedure_as_drug",
                    "severity": "warning",
                    "entity_text": e["text"],
                    "detail": f"contains procedure keyword {kw!r} but typed THUỐC",
                })
                break


def check_risk_factor_leak(entities, issues):
    for e in entities:
        text_lower = e["text"].lower()
        for kw in RISK_FACTOR_KEYWORDS:
            if kw in text_lower:
                issues.append({
                    "check": "suspected_risk_factor_leak",
                    "severity": "warning",
                    "entity_text": e["text"],
                    "detail": f"contains risk-factor keyword {kw!r}, typed {e['type']} — should likely not be extracted at all",
                })
                break


def _current_clause_start(raw_text: str, position: int) -> int:
    """Find where the clause containing `position` begins, by locating the
    nearest clause boundary before it (mirrors src/assertions/assertions.py
    so both use the same definition of 'clause')."""

    search_region = raw_text[:position]
    boundaries = [m.end() for m in CLAUSE_BOUNDARY.finditer(search_region)]

    return boundaries[-1] if boundaries else 0


def check_missing_assertions(entities, raw_text, issues):
    for e in entities:
        if e["type"] not in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"):
            continue

        start = e["position"][0]
        clause_start = _current_clause_start(raw_text, start)
        # Also bound by a max window so a very long clause doesn't pull in
        # an unrelated cue from far away.
        context_start = max(clause_start, start - CONTEXT_WINDOW)
        context = raw_text[context_start:start].lower()

        assertions = e.get("assertions", [])

        for cue in NEGATION_CUES:
            if re.search(r"\b" + re.escape(cue) + r"\b", context) and "isNegated" not in assertions:
                issues.append({
                    "check": "possible_missing_negation",
                    "severity": "info",
                    "entity_text": e["text"],
                    "detail": f"negation cue {cue!r} appears earlier in the same clause, but isNegated not set",
                })
                break

        for cue in FAMILY_CUES:
            if re.search(r"\b" + re.escape(cue) + r"\b", context) and "isFamily" not in assertions:
                issues.append({
                    "check": "possible_missing_family",
                    "severity": "info",
                    "entity_text": e["text"],
                    "detail": f"family cue {cue!r} appears earlier in the same clause, but isFamily not set",
                })
                break


def check_empty_candidates(entities, issues):
    for e in entities:
        if e["type"] not in ("CHẨN_ĐOÁN", "THUỐC"):
            continue

        # Redacted/masked text is EXPECTED to have empty candidates — not a
        # real issue, skip it.
        if re.fullmatch(r"\*+", e["text"].strip()):
            continue

        if not e.get("candidates"):
            issues.append({
                "check": "empty_candidates",
                "severity": "info",
                "entity_text": e["text"],
                "detail": f"{e['type']} entity has no candidates — possible retrieval miss or bad lookup_term",
            })


def validate_document(doc_id: str, entities: list[dict], raw_text: str) -> list[dict]:
    issues = []

    check_position_text_mismatch(entities, raw_text, issues)
    check_long_span(entities, issues)
    check_overlapping_spans(entities, issues)
    check_exact_duplicates(entities, issues)
    check_procedure_as_drug(entities, issues)
    check_risk_factor_leak(entities, issues)
    check_missing_assertions(entities, raw_text, issues)
    check_empty_candidates(entities, issues)

    for issue in issues:
        issue["doc_id"] = doc_id

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--input-dir", default="data/raw/input")
    parser.add_argument("--doc-id", default=None, help="Only validate this one doc")
    parser.add_argument("--report", default="validation_report.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)

    json_files = sorted(
        output_dir.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )

    if args.doc_id is not None:
        json_files = [f for f in json_files if f.stem == args.doc_id]

    if not json_files:
        print(f"No output files found in {output_dir} (doc-id filter: {args.doc_id})")
        return

    all_issues = []
    docs_checked = 0
    docs_missing_input = []

    for json_path in json_files:
        doc_id = json_path.stem
        input_path = input_dir / f"{doc_id}.txt"

        if not input_path.exists():
            docs_missing_input.append(doc_id)
            continue

        entities = load_entities(json_path)
        raw_text = unicodedata.normalize("NFC", input_path.read_text(encoding="utf-8"))

        issues = validate_document(doc_id, entities, raw_text)
        all_issues.extend(issues)
        docs_checked += 1

    # --- console summary ---
    by_check = defaultdict(list)
    for issue in all_issues:
        by_check[issue["check"]].append(issue)

    print(f"Checked {docs_checked} documents, found {len(all_issues)} flagged items.\n")

    if docs_missing_input:
        print(f"WARNING: {len(docs_missing_input)} output files had no matching input .txt "
              f"(skipped): {docs_missing_input[:10]}\n")

    severity_order = {"critical": 0, "warning": 1, "info": 2}

    for check_name, issues in sorted(by_check.items(), key=lambda kv: severity_order.get(kv[1][0]["severity"], 9)):
        severity = issues[0]["severity"]
        docs_affected = len(set(i["doc_id"] for i in issues))
        print(f"[{severity:8s}] {check_name:32s} {len(issues):4d} items across {docs_affected} docs")

    # --- detailed markdown report ---
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# Validation Report\n\n")
        f.write(f"Checked {docs_checked} documents, {len(all_issues)} flagged items.\n\n")

        for check_name, issues in sorted(by_check.items(), key=lambda kv: severity_order.get(kv[1][0]["severity"], 9)):
            f.write(f"## {check_name} ({len(issues)} items)\n\n")
            for issue in issues:
                f.write(f"- **doc {issue['doc_id']}** — `{issue['entity_text']}`\n")
                f.write(f"  {issue['detail']}\n")
            f.write("\n")

    print(f"\nFull detail written to {args.report}")


if __name__ == "__main__":
    main()