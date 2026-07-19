from dataclasses import dataclass

@dataclass
class Section:

    title: str

    text: str

    start: int

    end: int

    @property
    def length(self):
        return self.end - self.start

    def contains(self, position: int):
        return self.start <= position < self.end
