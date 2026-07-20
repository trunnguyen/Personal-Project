from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from src.llm.response_parser import parse_entities, ParseError
from src.preprocessing.span_locator import SpanLocator

from src.models.entity import Entity
from src.models.entity_type import EntityType
from src.models.assertion import Assertion
from src.models.section import Section


TYPE_MAP = {
    "TRIỆU_CHỨNG": EntityType.SYMPTOM,
    "TÊN_XÉT_NGHIỆM": EntityType.LAB_TEST,
    "KẾT_QUẢ_XÉT_NGHIỆM": EntityType.LAB_RESULT,
    "CHẨN_ĐOÁN": EntityType.DIAGNOSIS,
    "THUỐC": EntityType.DRUG,
}

ASSERTION_MAP = {
    "isNegated": Assertion.NEGATED,
    "isFamily": Assertion.FAMILY,
    "isHistorical": Assertion.HISTORICAL,
}


class LlmEntityExtractor:
    """
    Replaces the four separate rule-based extractors (drug/diagnosis/
    symptom/lab) with a single LLM call per section that extracts all 5
    entity types + assertions together, then grounds CHẨN_ĐOÁN/THUỐC
    candidates against the real ICD-10/RxNorm corpus via retrieval rather
    than trusting the model's own recollection of exact code numbers.

    Every LLM-proposed entity passes through a hallucination guard
    (SpanLocator) before becoming an Entity: if its "text" can't actually
    be located in the source, it's dropped rather than exported with a
    fabricated position.
    """

    def __init__(self, llm_client, icd_retriever, rxnorm_retriever, candidate_top_k: int = 3):

        self.llm_client = llm_client
        self.icd_retriever = icd_retriever
        self.rxnorm_retriever = rxnorm_retriever
        self.candidate_top_k = candidate_top_k

    def extract(self, section: Section, offset_map=None) -> list[Entity]:

        section_text = section.text

        if not section_text or not section_text.strip():
            return []

        raw_response = self.llm_client.chat(
            SYSTEM_PROMPT,
            build_user_prompt(section_text),
        )

        try:
            raw_entities = parse_entities(raw_response)
        except ParseError:
            # A malformed response for one section shouldn't take down the
            # whole document — skip this section's entities rather than
            # crash the run. (Worth logging in the real pipeline; kept
            # simple here.)
            return []

        locator = SpanLocator()

        entities = []

        for raw in raw_entities:

            span = locator.locate(section_text, raw["text"])

            if span is None:
                # Hallucination guard: model claimed text that doesn't
                # actually appear in the source. Drop it.
                continue

            rel_start, rel_end = span

            abs_start = section.start + rel_start
            abs_end = section.start + rel_end

            if offset_map is not None:
                abs_start = offset_map.original_index(abs_start)
                # Map the last INCLUDED character then make exclusive again,
                # since offset_map only has entries for indices < len(normalized).
                abs_end = offset_map.original_index(abs_end - 1) + 1

            entity = Entity(
                text=section_text[rel_start:rel_end],  # true verbatim substring
                start=abs_start,
                end=abs_end,
                entity_type=TYPE_MAP[raw["type"]],
                section=section,
            )

            entity.assertions = [
                ASSERTION_MAP[a] for a in raw["assertions"] if a in ASSERTION_MAP
            ]

            if raw["type"] == "CHẨN_ĐOÁN" and raw["lookup_term"]:
                entity.candidates = self.icd_retriever.search(
                    raw["lookup_term"], top_k=self.candidate_top_k
                )
            elif raw["type"] == "THUỐC" and raw["lookup_term"]:
                entity.candidates = self.rxnorm_retriever.search(
                    raw["lookup_term"], top_k=self.candidate_top_k
                )

            entities.append(entity)

        return entities
