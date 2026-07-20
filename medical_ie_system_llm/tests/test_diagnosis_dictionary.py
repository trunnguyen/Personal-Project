from src.knowledge.diagnosis_dictionary import DiagnosisDictionary
dictionary = DiagnosisDictionary()

print("Unique diagnoses:", len(dictionary.entries))

for name in list(dictionary.entries.keys())[:10]:

    print(dictionary.entries[name])