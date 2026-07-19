from src.models.document import Document

from src.extractions.drug_extractor import DrugExtractor
from src.extractions.symptom_extractor import SymptomExtractor
from src.extractions.diagnosis_extractor import DiagnosisExtractor
from src.extractions.lab_extractor import LabExtractor

class MedicalEntityExtractor:
    def __init__(self):
        self.extractor = [
            DrugExtractor(),
            SymptomExtractor(),
            DiagnosisExtractor(),
            LabExtractor(),
        ]

    def process(self, document: Document) -> Document:

        document.entities = []

        for section in document.sections:
            for extractor in self.extractor:
                entities = extractor.extract(section)
                document.entities.extend(entities)

        document.entities.sort(key=lambda entity: entity.start)

        return document