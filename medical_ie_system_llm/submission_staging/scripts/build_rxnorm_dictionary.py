from pathlib import Path
import csv
import re
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RXNCONSO_PATH = (
    PROJECT_ROOT /
    "data" /
    "knowledge" /
    "rxnorm" /
    "RXNCONSO.RRF"
)

OUTPUT_PATH = (
    PROJECT_ROOT /
    "data" /
    "knowledge" /
    "drugs.csv"
)


def build_dictionary():

    print("Reading RxNorm...")

    TTY_PRIORITY = {
        "SCD": 1,
        "SBD": 2,
        "GPCK": 3,
        "BPCK": 4,
        "SCDG": 5,
        "SBDG": 6,
        "SCDF": 7,
        "SBDF": 8,
        "IN": 9,
        "PIN": 10,
        "MIN": 11,
    }

    KEEP_TTYS = {
        # Ingredient
        "IN",
        "PIN",
        "MIN",

        # Clinical drugs
        "SCD",
        "SBD",

        # Generic/Brand packs
        "GPCK",
        "BPCK",

        # Dose groups
        "SCDG",
        "SBDG",

        # Dose forms
        "SCDF",
        "SBDF",
    }

    rows = []
    seen = set()

    with open(RXNCONSO_PATH, encoding="utf-8") as f:

        for line in f:

            cols = line.rstrip("\n").split("|")

            if len(cols) < 15:
                continue

            concept_id = cols[0]
            lat = cols[1]
            sab = cols[11]
            tty = cols[12]
            name = cols[14].lower().strip()

            name = re.sub(r"\s+", " ", name)

            name = (
                name.replace("milligram", "mg")
                .replace("milliliter", "ml")
                .replace("microgram", "mcg")
            )
            # Keep only English concepts
            if lat != "ENG":
                continue

            # Keep only RxNorm vocabulary
            if sab != "RXNORM":
                continue

            # Keep only useful drug types
            if tty not in KEEP_TTYS:
                continue

            # Remove duplicates
            key = (concept_id, name)

            if key in seen:
                continue

            seen.add(key)
            length =len(name.split())
            rows.append(
                {
                    "concept_id": concept_id,
                    "name": name,
                    "tty": tty,
                    "priority": TTY_PRIORITY[tty],
                    "length": length,

                }
            )

    counter = Counter(row["tty"] for row in rows)

    print("\nTTY distribution:")

    for tty, count in counter.most_common():
        print(f"{tty:5} {count}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concept_id",
                "name",
                "tty",
                "priority",
                "length",
            ],
        )

        writer.writeheader()

        writer.writerows(rows)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":

    build_dictionary()