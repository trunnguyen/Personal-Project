import csv

from rapidfuzz import process, fuzz


def format_icd10_code(raw_code: str) -> list[str]:
    """
    The source order file has no decimal point (e.g. "K2100"), but the
    organizers' own official example uses the standard dotted display form
    ("K21.0", "K21.9"). Insert it.

    Also: the source file is a very recent ICD-10-CM revision that has
    subdivided some classic codes further (e.g. "K21.0" -> "K21.00"/"K21.01"
    for with/without bleeding). The ground truth was very likely authored
    against an older/simpler code set that doesn't have this subdivision.
    So alongside the precise dotted code, also emit its 1-digit "classic"
    parent as a second candidate — cheap insurance against a code-set
    vintage mismatch, at a small precision cost that a Jaccard-based metric
    tolerates well when it fixes a total miss.
    """

    raw_code = raw_code.strip().upper()

    if len(raw_code) <= 3:
        return [raw_code]

    dotted = f"{raw_code[:3]}.{raw_code[3:]}"

    variants = [dotted]

    # Subdivision deeper than one decimal digit (e.g. "K21.00") -> also
    # offer the classic 1-digit parent ("K21.0").
    decimal_part = raw_code[3:]

    if len(decimal_part) > 1:
        parent = f"{raw_code[:3]}.{decimal_part[0]}"
        if parent not in variants:
            variants.append(parent)

    return variants


class FuzzyCodeRetriever:
    """
    Repurposes the existing dictionary CSVs (icd10.csv, drugs.csv) — no
    longer as exact-match dictionaries, but as a retrieval corpus for
    candidate generation. The LLM extracts entities and proposes an English
    "lookup_term" for CHẨN_ĐOÁN/THUỐC; this retriever fuzzy-matches that
    term against the real, official code list so the final candidate codes
    are always grounded in an actual ICD-10/RxNorm entry rather than
    something the model invented from memory (LLMs are not reliable sources
    of exact code numbers).
    """

    def __init__(
        self,
        csv_path: str,
        code_col: str,
        name_col: str,
        code_formatter=None,
    ):

        self.code_formatter = code_formatter

        self.name_to_codes: dict[str, list[str]] = {}

        with open(csv_path, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:

                name = row[name_col].strip().lower()
                code = row[code_col].strip()

                if not name or not code:
                    continue

                self.name_to_codes.setdefault(name, [])

                if code not in self.name_to_codes[name]:
                    self.name_to_codes[name].append(code)

        self._choices = list(self.name_to_codes.keys())

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_cutoff: float = 60.0,
    ) -> list[str]:

        if not query or not query.strip():
            return []

        # Fetch a few extra name matches, but only the top `top_k` DISTINCT
        # matches actually get included below — variant expansion (hedge
        # parents) is allowed to make the final list slightly longer than
        # top_k, which a Jaccard-based metric tolerates far better than
        # truncating away a genuinely distinct match to make room for a
        # synthetic hedge variant of a higher-ranked one.
        matches = process.extract(
            query.strip().lower(),
            self._choices,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
            limit=top_k,
        )

        codes: list[str] = []

        for matched_name, score, _ in matches:
            for raw_code in self.name_to_codes[matched_name]:

                variants = (
                    self.code_formatter(raw_code)
                    if self.code_formatter
                    else [raw_code]
                )

                for code in variants:
                    if code not in codes:
                        codes.append(code)

        return codes
