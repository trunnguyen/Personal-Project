from src.models.document import Document


class LlmMedicalEntityExtractor:
    """
    Replaces MedicalEntityExtractor (which looped 4 separate rule-based
    extractors) — now a single LLM-backed extractor per section. Also
    applies document.offset_map so exported positions are relative to the
    ORIGINAL raw input file, not the whitespace-normalized text sections
    are built from (offset_map existed before but was never actually wired
    into the extraction path).
    """

    def __init__(self, extractor):
        self.extractor = extractor

    def process(self, document: Document) -> Document:

        document.entities = []

        for section in document.sections:

            entities = self.extractor.extract(section, document.offset_map)

            document.entities.extend(entities)

        document.entities.sort(key=lambda entity: entity.start)

        return document
