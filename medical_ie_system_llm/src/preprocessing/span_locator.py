import re


class SpanLocator:
    """
    Locates an LLM-extracted "text" span inside the real section text, and
    returns its (start, end) offset. This is the hallucination guard: if the
    text doesn't actually appear (the model paraphrased, invented, or
    mis-transcribed it), locate() returns None and the caller must drop the
    entity rather than emit a fabricated position.

    Tolerant of whitespace-run differences (single vs collapsed spaces) and
    case differences between what the model echoed and the source, since
    those are cosmetic — but the actual token content must match exactly.

    Stateful per section: tracks a search cursor per distinct entity text so
    that repeated mentions (e.g. "táo bón" appearing twice, tied to two
    different drugs, as in the official example) resolve to successive
    occurrences left-to-right rather than all collapsing onto the first one.
    """

    def __init__(self):
        self._cursor: dict[str, int] = {}

    def reset(self):
        self._cursor = {}

    def locate(self, section_text: str, entity_text: str):

        pattern = self._build_pattern(entity_text)

        if pattern is None:
            return None

        start_from = self._cursor.get(entity_text, 0)

        match = pattern.search(section_text, start_from)

        if match is None and start_from > 0:
            # Model may not have emitted entities in strict left-to-right
            # order — retry from the beginning before giving up.
            match = pattern.search(section_text, 0)

        if match is None:
            return None

        self._cursor[entity_text] = match.end()

        return match.start(), match.end()

    @staticmethod
    def _build_pattern(entity_text: str):

        tokens = entity_text.split()

        if not tokens:
            return None

        # Join tokens with a flexible whitespace pattern so a difference in
        # spacing (e.g. double space in source, single in model output)
        # doesn't cause a false "hallucination" rejection.
        pattern_str = r"\s+".join(re.escape(t) for t in tokens)

        try:
            return re.compile(pattern_str, re.IGNORECASE)
        except re.error:
            return None
