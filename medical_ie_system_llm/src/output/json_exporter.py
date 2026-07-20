import json

from src.models.entity import Entity
from src.models.entity_type import EntityType


ASSERTION_ELIGIBLE_TYPES = {
    EntityType.SYMPTOM,
    EntityType.DIAGNOSIS,
    EntityType.DRUG,
}

CANDIDATE_ELIGIBLE_TYPES = {
    EntityType.DIAGNOSIS,
    EntityType.DRUG,
}


class JsonExporter:

    def export(self, entities: list[Entity], output_path: str):

        output = []

        for entity in entities:

            item = {

                "text": entity.text,

                "type": entity.entity_type.value,

                "position": [
                    entity.start,
                    entity.end,
                ],
            }

            # Match the official example schema exactly: assertions/
            # candidates keys are only present for the entity types the
            # spec defines them for (TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM
            # have neither), rather than always emitting an empty list.
            if entity.entity_type in ASSERTION_ELIGIBLE_TYPES:
                item["assertions"] = [
                    assertion.value
                    for assertion in entity.assertions
                ]

            if entity.entity_type in CANDIDATE_ELIGIBLE_TYPES:
                item["candidates"] = entity.candidates

            output.append(item)

        with open(output_path, "w", encoding="utf-8") as f:

            json.dump(
                output,
                f,
                ensure_ascii=False,
                indent=2,
            )