from app.config import Config

from src.knowledge.symptom_dictionary import SymptomDictionary
from src.models.symptom_match import SymptomMatch


class SymptomMatcher:

    def __init__(self):

        config = Config()

        self.dictionary = SymptomDictionary(
            config.symptom_dictionary_path
        )

    def match(self, text: str):

        matches = []

        text_lower = text.lower()

        for symptom_name in self.dictionary.entries:

            start = text_lower.find(symptom_name)

            while start != -1:

                end = start + len(symptom_name)

                matches.append(
                    SymptomMatch(
                        text=text[start:end],
                        start=start,
                        end=end,
                        concepts=[],
                    )
                )

                start = text_lower.find(
                    symptom_name,
                    start + 1,
                )

        matches.sort(key=lambda x: x.start)

        return matches