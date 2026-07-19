from abc import ABC, abstractmethod

from src.models.section import Section
from src.models.entity import Entity

class BaseExtractor(ABC):

    @abstractmethod
    def extract(self,section: Section,) -> list[Entity]:
        pass