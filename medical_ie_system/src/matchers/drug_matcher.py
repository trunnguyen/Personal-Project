import re

from src.knowledge.drug_dictionary import DrugDictionary
from src.models.drug_match import DrugMatch


class DrugMatcher:

    def __init__(self):

        self.dictionary = DrugDictionary()

    def match(self, text: str):

        matches = []

        for name, concepts in self.dictionary.index.items():

            pattern = r"\b" + re.escape(name) + r"\b"

            for m in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                matches.append(

                    DrugMatch(

                        text=m.group(),

                        start=m.start(),

                        end=m.end(),

                        concepts=concepts,
                    )

                )

        return matches