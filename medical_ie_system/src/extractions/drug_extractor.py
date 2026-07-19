import re

from src.extractions.base_extractor import BaseExtractor
from src.extractions.patterns import DRUG_PATTERN
from src.models.entity import Entity
from src.models.section import Section
from src.utils.text import split_sentences

class DrugExtractor(BaseExtractor):

    def extract(self,section: Section) -> list[Entity]:
        entities = []

        section_start = section.start

        current_offset = 0

        sentences = split_sentences(section.text)
        for sentence in sentences:
            sentence_start = section.text.find(sentence, current_offset)

            if sentence_start == -1:
                continue

            current_offset = sentence_start + len(sentence)

            for match in DRUG_PATTERN.finditer(sentence):
                text = match.group().strip()

                start = section_start + sentence_start + match.start()

                end = section_start + sentence_start + match.end()

                entity = Entity(
                    text=text,
                    entity_type=EntityType.DRUG,
                    start=start,
                    end=end,
                    section=section,)
                entities.append(entity)
        return entities