import csv
from collections import defaultdict
from pathlib import Path

from src.models.drug_concept import DrugConcept


class DrugDictionary:

    def __init__(self):

        path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "knowledge"
            / "drugs.csv"
        )

        self.index = defaultdict(list)

        with open(path, encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:

                concept = DrugConcept(
                    concept_id=row["concept_id"],
                    name=row["name"],
                    tty=row["tty"],
                )

                self.index[row["name"]].append(concept)

    def lookup(self, text: str):
        return self.index.get(text.lower(), [])

    def __len__(self):
        return len(self.index)