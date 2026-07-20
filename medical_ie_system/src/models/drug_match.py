from dataclasses import dataclass

from src.models.drug_concept import DrugConcept

@dataclass(slots=True)
class DrugMatch:

    text: str

    start: int

    end: int

    concepts: list[DrugConcept]