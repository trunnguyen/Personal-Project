from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from src.llm.response_parser import parse_entities, ParseError
from src.preprocessing.span_locator import SpanLocator

from src.models.entity import Entity
from src.models.entity_type import EntityType
from src.models.assertion import Assertion
from src.models.document import Document


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
    One LLM call PER DOCUMENT. Every LLM-proposed entity passes through a
    hallucination guard (SpanLocator) before becoming an Entity: if its
    "text" can't actually be located in the source, it's dropped rather
    than exported with a fabricated position. CHẨN_ĐOÁN/THUỐC candidates
    are grounded via retrieval against the real ICD-10/RxNorm corpus,
    never trusted from the model's own memory of exact code numbers.
    """

    def __init__(self, llm_client, icd_retriever, rxnorm_retriever, candidate_top_k: int = 3):

        self.llm_client = llm_client
        self.icd_retriever = icd_retriever
        self.rxnorm_retriever = rxnorm_retriever
        self.candidate_top_k = candidate_top_k

    def extract_document(self, document: Document) -> list[Entity]:

        document_text = document.normalized_text

        if not document_text or not document_text.strip():
            return []

        raw_response = self.llm_client.chat(
            SYSTEM_PROMPT,
            build_user_prompt(document_text),
        )

        try:
            raw_entities = parse_entities(raw_response)
        except ParseError as e:
            print(f"  [llm] PARSE FAILURE: {e}")
            print(f"  [llm] raw response was:\n{raw_response[:3000]}")
            return []

        if not raw_entities:
            # json.loads succeeded but yielded nothing usable — either the
            # model genuinely said "no entities" ([]), or every item got
            # filtered out by parse_entities' own validation (bad "type",
            # missing "text", etc). Print the raw response either way,
            # since silently returning [] here is exactly as undebuggable
            # as the earlier ParseError case was.
            print(f"  [llm] parsed to ZERO usable entities. Raw response was:\n{raw_response[:3000]}")
            return []

        locator = SpanLocator()

        entities = []
        rejected = []

        for raw in raw_entities:

            span = locator.locate(document_text, raw["text"])

            if span is None:
                # Hallucination guard: model claimed text that doesn't
                # actually appear in the source. Drop it — but track it so
                # we can see if this guard is the thing eating everything.
                rejected.append(raw["text"])
                continue

            rel_start, rel_end = span

            section = self._find_section(document.sections, rel_start)

            if section is None:
                continue

            abs_start, abs_end = rel_start, rel_end

            if document.offset_map is not None:
                abs_start = document.offset_map.original_index(abs_start)
                abs_end = document.offset_map.original_index(abs_end - 1) + 1

            entity = Entity(
                text=document_text[rel_start:rel_end],
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

        if rejected:
            print(f"  [llm] {len(rejected)}/{len(raw_entities)} entities REJECTED by hallucination guard (text not found verbatim in source):")
            for text in rejected[:10]:
                print(f"    - {text!r}")

        return entities

    @staticmethod
    def _find_section(sections, position: int):

        for section in sections:
            if section.start <= position < section.end:
                return section

        return sections[-1] if sections else None