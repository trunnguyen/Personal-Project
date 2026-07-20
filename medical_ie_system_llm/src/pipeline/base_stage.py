from abc import ABC, abstractmethod

from src.models.document import Document

class PipelineStage(ABC):
    @abstractmethod
    def process(self, document: Document) -> Document:
        ...