from app.config import Config

from src.knowledge.diagnosis_dictionary import DiagnosisDictionary
from src.knowledge.diagnosis_match import DiagnosisMatch


class DiagnosisMatcher:

    def __init__(self):

        config = Config()

        self.dictionary = DiagnosisDictionary()

    def match(self, text: str):

        matches = []

        text_lower = text.lower()

        for diagnosis_name, concept in self.dictionary.entries.items():

            start = text_lower.find(diagnosis_name)

            while start != -1:

                end = start + len(diagnosis_name)

                matches.append(

                    DiagnosisMatch(

                        text=text[start:end],

                        start=start,

                        end=end,

                        concepts=[concept],
                    )
                )

                start = text_lower.find(
                    diagnosis_name,
                    start + 1,
                )

        return matches