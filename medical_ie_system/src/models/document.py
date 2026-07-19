from dataclasses import dataclass, field
from pathlib import Path

from src.models.offset_map import OffsetMap
from src.models.section import Section

@dataclass
class Document:

    doc_id: str

    file_path: Path

    text: str

    normalized_text: str = ""

    offset_map: OffsetMap | None =None

    sections: list[Section] = field(default_factory=list)

    entities: list = field(default_factory=list)