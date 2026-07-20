from src.extractions.base_extractor import BaseExtractor
from src.matchers.diagnosis_matcher import DiagnosisMatcher

from src.models.entity import Entity
from src.models.entity_type import EntityType

from src.assertions.assertions import AssertionDetector


class DiagnosisExtractor(BaseExtractor):

    def __init__(self):

        self.matcher = DiagnosisMatcher()
        self.assertion_detector = AssertionDetector()

    def extract(self, section):

        entities = []

        matches = self.matcher.match(section.text)

        for match in matches:

            entity = Entity(

                text=match.text,

                start=section.start + match.start,

                end=section.start + match.end,

                entity_type=EntityType.DIAGNOSIS,

                section=section,
            )

            entity.candidates = [

                c.code

                for c in match.concepts
            ]

            entity.assertions = self.assertion_detector.detect(entity)

            entities.append(entity)

        return entities
