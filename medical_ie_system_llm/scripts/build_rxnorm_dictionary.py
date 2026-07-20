from pathlib import Path
import csv


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

    KEEP_TTYS = {
        "IN",  # Ingredient
        "PIN",  # Precise Ingredient
        "MIN",  # Multiple Ingredient
        "SCD",  # Semantic Clinical Drug
        "SBD",  # Semantic Branded Drug
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
            name = cols[14].strip().lower()

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

            rows.append(
                {
                    "concept_id": concept_id,
                    "name": name,
                    "tty": tty,
                }
            )

    print(f"Collected {len(rows):,} concepts")

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
            ],
        )

        writer.writeheader()

        writer.writerows(rows)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":

    build_dictionary()