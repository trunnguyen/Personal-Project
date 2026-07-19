from pathlib import Path
import csv

from src.knowledge.base_dictionary import BaseDictionary
from src.knowledge.dictionary_match import DictionaryMatch


class DrugDictionary(BaseDictionary):

    def __init__(self, csv_path: str = "data/knowledge/drugs.csv"):

        self.entries = []

        with open(csv_path, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:

                self.entries.append(
                    {
                        "concept_id": row["concept_id"],
                        "name": row["name"].strip(),
                        "name_lower": row["name"].strip().lower(),
                    }
                )

    def search(self, text: str) -> list[DictionaryMatch]:

        matches = []

        text_lower = text.lower()

        for entry in self.entries:

            start = 0

            while True:

                idx = text_lower.find(entry["name_lower"], start)

                if idx == -1:
                    break

                matches.append(
                    DictionaryMatch(
                        text=text[idx: idx + len(entry["name"])],
                        start=idx,
                        end=idx + len(entry["name"]),
                        concept_id=entry["concept_id"],
                    )
                )

                start = idx + len(entry["name"])

        return matches