from src.matchers.drug_matcher import DrugMatcher

from src.models.entity import Entity
from src.models.entity_type import EntityType

from src.assertions.assertions import AssertionDetector


class DrugExtractor:

    def __init__(self):

        self.matcher = DrugMatcher()
        self.assertion_detector = AssertionDetector()

    def extract(self, section):

        entities = []

        matches = self.matcher.match(section.text)

        for match in matches:

            entity = Entity(

                text=match.text,

                start=section.start + match.start,

                end=section.start + match.end,

                entity_type=EntityType.DRUG,

                section=section,
            )

            entity.candidates = [

                c.concept_id

                for c in match.concepts
            ]

            entity.assertions = self.assertion_detector.detect(entity)

            entities.append(entity)

        return entities
