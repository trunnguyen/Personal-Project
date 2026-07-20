import re

from src.models.assertion import Assertion

# Keyword-based, not exact-match: real section titles vary a lot
# ("Tiền sử bệnh", "Tiền sử bệnh nội khoa", "Tiền sử bệnh hiện tại", ...)
HISTORICAL_TITLE_KEYWORDS = [
    "tiền sử",
    "tiền căn",
    "thuốc trước khi nhập viện",
    "thuốc trước nhập viện",
    "trước khi nhập viện",
]

# "Tiền sử bệnh hiện tại" ("history of PRESENT illness") contains "tiền sử"
# as a naming idiom but describes the current complaint, not past history —
# exclude titles qualified as "hiện tại" (current/present).
HISTORICAL_TITLE_EXCLUDE = [
    "hiện tại",
]

NEGATION_CUES = [
    "không ghi nhận",
    "không có",
    "chưa ghi nhận",
    "phủ nhận",
    "loại trừ",
    "không",
    "chưa",
]

FAMILY_CUES = [
    "gia đình",
    "người nhà",
    "họ hàng",
    "tiền sử gia đình",
    "bố bệnh nhân",
    "mẹ bệnh nhân",
    "cha bệnh nhân",
    "anh bệnh nhân",
    "chị bệnh nhân",
    "em bệnh nhân",
    "con bệnh nhân",
    "ông bệnh nhân",
    "bà bệnh nhân",
]

# Boundaries that end a clause: newline, bullet dash at line start,
# sentence-ending punctuation, semicolon. NOT commas, since a Vietnamese
# negation like "Không buồn nôn, hay nôn, đổ mồ hôi" scopes over the
# whole comma-separated list.
CLAUSE_BOUNDARY = re.compile(r"[\n;.]|(?<=\n)\s*-")


class AssertionDetector:

    def detect(self, entity):

        assertions = []

        title = entity.section.title.lower()

        is_historical_title = (
            any(keyword in title for keyword in HISTORICAL_TITLE_KEYWORDS)
            and not any(exclude in title for exclude in HISTORICAL_TITLE_EXCLUDE)
        )

        if is_historical_title:
            assertions.append(Assertion.HISTORICAL)

        clause, clause_start = self._current_clause(entity)

        # Only look at the part of the clause BEFORE the entity —
        # cues that come after don't govern this mention.
        rel_start = entity.start - entity.section.start - clause_start
        scope = clause[:max(rel_start, 0)].lower()

        if self._contains_cue(scope, NEGATION_CUES):
            assertions.append(Assertion.NEGATED)

        if self._contains_cue(scope, FAMILY_CUES):
            assertions.append(Assertion.FAMILY)

        return assertions

    def _current_clause(self, entity):

        text = entity.section.text

        rel_start = entity.start - entity.section.start

        boundaries = [0] + [m.start() for m in CLAUSE_BOUNDARY.finditer(text)] + [len(text)]

        clause_start = 0

        for b in boundaries:
            if b <= rel_start:
                clause_start = b
            else:
                clause_end = b
                break
        else:
            clause_end = len(text)

        return text[clause_start:clause_end], clause_start

    def _contains_cue(self, scope: str, cues: list[str]) -> bool:

        for cue in cues:
            if re.search(r"\b" + re.escape(cue) + r"\b", scope):
                return True

        return False