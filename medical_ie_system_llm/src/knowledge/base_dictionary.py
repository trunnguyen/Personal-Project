from abc import ABC, abstractmethod

from src.knowledge.dictionary_match import DictionaryMatch


class BaseDictionary(ABC):

    @abstractmethod
    def search(self, text: str) -> list[DictionaryMatch]:
        pass