from dataclasses import dataclass

@dataclass
class OffsetMap:
    mapping: list[int]

    def original_index(self, normalized_index: int) -> int:
        return self.mapping[normalized_index]

    def __len__(self):
        return len(self.mapping)