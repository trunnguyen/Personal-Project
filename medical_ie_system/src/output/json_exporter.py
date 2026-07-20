import json

from src.models.entity import Entity


class JsonExporter:

    def export(self, entities: list[Entity], output_path: str):

        output = []

        for entity in entities:

            output.append({

                "text": entity.text,

                "type": entity.entity_type.value,

                "candidates": entity.candidates,

                "assertions": [
                    assertion.value
                    for assertion in entity.assertions
                ],

                "position": [
                    entity.start,
                    entity.end,
                ],
            })

        with open(output_path, "w", encoding="utf-8") as f:

            json.dump(
                output,
                f,
                ensure_ascii=False,
                indent=2,
            )