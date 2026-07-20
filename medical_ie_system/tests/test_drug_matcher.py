from src.matchers.drug_matcher import DrugMatcher

matcher = DrugMatcher()

text = """
Bệnh nhân được dùng aspirin 81 mg,
metoprolol 25mg po bid
và acetaminophen khi sốt.
"""

matches = matcher.find(text)

for match in matches:

    print(match)