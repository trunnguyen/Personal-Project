from dataclasses import dataclass


@dataclass
class DiagnosisMatch:

    text: str

    start: int

    end: int

    concepts: list