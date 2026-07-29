import re

DOSAGE = r"\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*(?:mg|mcg|g|ml|IU|%)"

ROUTE = r"(?:po|iv|im|sc|pr|sl|topical|oral)"

FREQUENCY = r"(?:bid|tid|qid|qam|qhs|q\d+h|daily|prn)"

DRUG_PATTERN = re.compile(
    rf"""
    (?P<drug>[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\- ]+?)
    (?:\s+(?P<dose>{DOSAGE}))?
    (?:\s+(?P<route>{ROUTE}))?
    (?:\s+(?P<freq>{FREQUENCY}))?
    """,
    re.IGNORECASE | re.VERBOSE,
)