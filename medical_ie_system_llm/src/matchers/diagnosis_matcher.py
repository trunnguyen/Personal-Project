import ahocorasick

from app.config import Config

from src.knowledge.diagnosis_dictionary import DiagnosisDictionary
from src.knowledge.diagnosis_match import DiagnosisMatch


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


class DiagnosisMatcher:

    def __init__(self):

        config = Config()

        self.dictionary = DiagnosisDictionary()

        # Same Aho-Corasick fix as DrugMatcher — this dictionary has
        # ~75k ICD-10 entries, which made both the per-entry loop and the
        # combined-regex approach too slow to be usable.
        self._automaton = ahocorasick.Automaton()

        for name in self.dictionary.entries.keys():
            self._automaton.add_word(name, name)

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

            concept = self.dictionary.entries.get(name)

            if concept is None:
                continue

            matches.append(

                DiagnosisMatch(

                    text=text[start_index:end_index + 1],

                    start=start_index,

                    end=end_index + 1,

                    concepts=[concept],
                )
            )

        matches.sort(key=lambda m: m.start)

        return matches
