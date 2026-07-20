import csv

from app.config import Config
from src.models.diagnosis_concept import DiagnosisConcept


class DiagnosisDictionary:

    def __init__(self):

        config = Config()

        self.entries = {}

        with open(
            config.diagnosis_dictionary_path,
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                concept = DiagnosisConcept(

                    code=row["code"],

                    name=row["name"],
                )

                self.entries[
                    concept.name.lower()
                ] = concept