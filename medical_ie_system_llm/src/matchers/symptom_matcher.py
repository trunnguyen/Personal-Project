import ahocorasick

from app.config import Config

from src.knowledge.symptom_dictionary import SymptomDictionary
from src.models.symptom_match import SymptomMatch


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


class SymptomMatcher:

    def __init__(self):

        config = Config()

        self.dictionary = SymptomDictionary(
            config.symptom_dictionary_path
        )

        # Same Aho-Corasick approach as DrugMatcher/DiagnosisMatcher, ready
        # for whenever the symptom dictionary is populated. Also fixes a
        # real bug in the old .find()-based version: it had no word-boundary
        # check, so a short symptom term like "ho" (cough) would match
        # inside an unrelated word (e.g. "cholesterol").
        self._automaton = ahocorasick.Automaton()

        for symptom_name in self.dictionary.entries:
            self._automaton.add_word(symptom_name, symptom_name)

        self._automaton.make_automaton()

    def match(self, text: str):

        text_lower = text.lower()

        matches = []

        for end_index, name in self._automaton.iter(text_lower):

            start_index = end_index - len(name) + 1

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

            matches.append(
                SymptomMatch(
                    text=text[start_index:end_index + 1],
                    start=start_index,
                    end=end_index + 1,
                    concepts=[],
                )
            )

        matches.sort(key=lambda x: x.start)

        return matches
