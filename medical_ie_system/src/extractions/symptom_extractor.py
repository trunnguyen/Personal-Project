from src.matchers.symptom_matcher import SymptomMatcher

from src.models.entity import Entity
from src.models.entity_type import EntityType

from src.assertions.assertions import AssertionDetector


class SymptomExtractor:

    def __init__(self):

        self.matcher = SymptomMatcher()

        self.assertion_detector = AssertionDetector()

    def extract(self, section):

        entities = []

        matches = self.matcher.match(section.text)

        for match in matches:

            entity = Entity(

                text=match.text,

                start=section.start + match.start,

                end=section.start + match.end,

                entity_type=EntityType.SYMPTOM,

                section=section,
            )

            # Symptoms have no candidate IDs
            entity.candidates = []

            entity.assertions = self.assertion_detector.detect(entity)

            entities.append(entity)

        return entities