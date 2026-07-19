from dataclasses import dataclass


@dataclass(slots=True)
class DictionaryMatch:
    text: str

    start: int

    end: int

    concept_id: str