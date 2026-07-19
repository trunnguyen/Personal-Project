from src.models.document import Document

from src.extraction.base_extractor import BaseEntityExtractor

class MedicalEntityExtractor(BaseEntityExtractor):
    def process(self, document: Document) -> Document:

        return document