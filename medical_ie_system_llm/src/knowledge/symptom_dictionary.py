import csv


class SymptomDictionary:

    def __init__(self, path):

        self.entries = {}

        with open(path, encoding="utf-8") as f:

            reader = csv.reader(f)

            for row in reader:

                if not row:
                    continue

                name = row[0].strip().lower()

                self.entries[name] = name