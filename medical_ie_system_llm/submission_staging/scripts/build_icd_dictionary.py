import csv
from pathlib import Path

SOURCE = Path("data/knowledge/icd10/icd10cm-codes-2027.txt")
OUTPUT = Path("data/knowledge/icd10.csv")

print("Reading ICD-10...")

count = 0

with open(SOURCE, encoding="utf-8") as infile, \
     open(OUTPUT, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile)

    # header
    writer.writerow(["code", "name"])

    for line in infile:

        line = line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            continue

        code, name = parts

        writer.writerow([
            code,
            name.lower()
        ])

        count += 1

print(f"Saved {count:,} ICD concepts")
print(f"Output: {OUTPUT}")
