from src.knowledge.drug_dictionary import DrugDictionary

dictionary = DrugDictionary()

print(f"Loaded {len(dictionary.entries)} drug entries")

text = """
Bệnh nhân được dùng aspirin 325mg và metoprolol 25mg po bid.
"""

matches = dictionary.search(text)

print(matches)

for match in matches:
    print(match)