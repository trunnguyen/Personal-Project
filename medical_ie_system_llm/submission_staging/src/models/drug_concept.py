from dataclasses import dataclass

@dataclass
class DrugConcept:
    concept_id: str
    name: str
    tty: str