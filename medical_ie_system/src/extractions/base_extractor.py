from abc import ABC, abstractmethod

from src.models.document import Document

from src.pipeline.base_stage import PipelineStage

class BaseEntityExtractor(PipelineStage,ABC):

    @abstractmethod
    def process(self, document: Document) -> Document:
        ...