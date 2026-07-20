from src.knowledge.drug_dictionary import DrugDictionary

dictionary = DrugDictionary()

print(f"Unique drug names: {len(dictionary)}")

for drug in [
    "aspirin",
    "metoprolol",
    "amlodipine",
    "acetaminophen",
]:
    print("=" * 40)
    print(drug)

    concepts = dictionary.lookup(drug)

    for concept in concepts:
        print(concept)