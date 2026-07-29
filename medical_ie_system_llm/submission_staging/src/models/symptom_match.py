from dataclasses import dataclass


@dataclass
class SymptomMatch:

    text: str

    start: int

    end: int

    concepts: list