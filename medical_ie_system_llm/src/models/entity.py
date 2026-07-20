from dataclasses import dataclass, field

from src.models.entity_type import EntityType
from src.models.candidate import Candidate
from src.models.assertion import Assertion
from src.models.section import Section
@dataclass
class Entity:

    text: str

    start: int

    end: int

    entity_type: EntityType

    section: Section

    assertions: list[Assertion] = field(default_factory=list)

    candidates: list[str] = field(default_factory=list)