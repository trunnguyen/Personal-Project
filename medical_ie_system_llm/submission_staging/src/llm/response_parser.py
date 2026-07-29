import json
import re


ALLOWED_TYPES = {
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
}

ALLOWED_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}

# Only these types carry assertions/candidates per the task spec.
ASSERTION_ELIGIBLE_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
CANDIDATE_ELIGIBLE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}


class ParseError(Exception):
    pass


def _extract_json_array(raw: str) -> str:
    """
    Models occasionally wrap output in ```json fences or add a stray
    sentence despite instructions not to. Pull out the first [...] block.
    """

    raw = raw.strip()

    # Strip markdown code fences if present.
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    if raw.startswith("[") and raw.endswith("]"):
        return raw

    # Fall back to locating the first top-level array in the text.
    start = raw.find("[")
    end = raw.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ParseError(f"No JSON array found in LLM response: {raw[:200]!r}")

    return raw[start:end + 1]


def parse_entities(raw_response: str) -> list[dict]:
    """
    Parse and validate raw LLM output into a list of entity dicts.
    Malformed individual entries are dropped (logged via return metadata
    would be nicer, but keep it simple: skip bad entries rather than
    failing the whole section on one bad object).
    """

    json_text = _extract_json_array(raw_response)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON from LLM: {e}\nRaw: {json_text[:300]!r}") from e

    if not isinstance(parsed, list):
        raise ParseError(f"Expected a JSON array, got {type(parsed)}")

    valid_entities = []

    for item in parsed:

        if not isinstance(item, dict):
            continue

        text = item.get("text")
        entity_type = item.get("type")

        if not isinstance(text, str) or not text.strip():
            continue

        if entity_type not in ALLOWED_TYPES:
            continue

        assertions = item.get("assertions") or []
        if not isinstance(assertions, list):
            assertions = []

        assertions = [a for a in assertions if a in ALLOWED_ASSERTIONS]

        if entity_type not in ASSERTION_ELIGIBLE_TYPES:
            assertions = []

        lookup_term = item.get("lookup_term")
        if entity_type not in CANDIDATE_ELIGIBLE_TYPES:
            lookup_term = None
        elif isinstance(lookup_term, str) and not lookup_term.strip():
            lookup_term = None

        valid_entities.append({
            "text": text,
            "type": entity_type,
            "assertions": assertions,
            "lookup_term": lookup_term,
        })

    return valid_entities
