import ahocorasick

from src.knowledge.drug_dictionary import DrugDictionary
from src.models.drug_match import DrugMatch


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


class DrugMatcher:

    def __init__(self):

        self.dictionary = DrugDictionary()

        # Aho-Corasick: matches all ~29k dictionary entries in a single
        # linear pass over the text, regardless of dictionary size.
        # (Previously: one regex search per dictionary entry — ~2s per
        # section. A single combined regex alternation was tried next but
        # Python's re engine still walks alternatives linearly, so it
        # didn't scale to the larger ICD-10 dictionary. AC does.)
        self._automaton = ahocorasick.Automaton()

        for name in self.dictionary.index.keys():
            self._automaton.add_word(name, name)

        self._automaton.make_automaton()

    def match(self, text: str):

        text_lower = text.lower()

        matches = []

        for end_index, name in self._automaton.iter(text_lower):

            start_index = end_index - len(name) + 1

            # Enforce word boundaries (AC alone would also match "aspirin"
            # inside a longer unrelated word).
            before_ok = (
                start_index == 0
                or not _is_word_char(text_lower[start_index - 1])
            )

            after_ok = (
                end_index + 1 == len(text_lower)
                or not _is_word_char(text_lower[end_index + 1])
            )

            if not (before_ok and after_ok):
                continue

            concepts = self.dictionary.index.get(name)

            if not concepts:
                continue

            matches.append(

                DrugMatch(

                    text=text[start_index:end_index + 1],

                    start=start_index,

                    end=end_index + 1,

                    concepts=concepts,
                )

            )

        matches.sort(key=lambda m: m.start)

        return matches
