from src.models.document import Document


class LlmMedicalEntityExtractor:
    """
    Makes exactly ONE LLM call per document (see
    LlmEntityExtractor.extract_document), not one per section.
    """

    def __init__(self, extractor):
        self.extractor = extractor

    def process(self, document: Document) -> Document:

        document.entities = self.extractor.extract_document(document)

        document.entities.sort(key=lambda entity: entity.start)

        return document