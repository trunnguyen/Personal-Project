from src.models.document import Document
from src.preprocessing.text_normalizer import DocumentNormalizer
from src.preprocessing.section_splitter import SectionSplitter
from src.extractions.medical_entity_extractor import MedicalEntityExtractor

class MedicalIEPipeline:
    def __init__(self, stages=None):
        if stages is None:
            stages = [
                DocumentNormalizer(),
                SectionSplitter(),
                MedicalEntityExtractor(),
            ]
        self.stages = stages

    def process(self, document: Document) -> Document:
        for stage in self.stages:
            document = stage.process(document)
        return document
