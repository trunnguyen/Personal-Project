from src.matchers.diagnosis_matcher import DiagnosisMatcher

matcher = DiagnosisMatcher()

text = """
The patient has hypertension and diabetes mellitus.
"""

matches = matcher.match(text)

for match in matches:
    print(match)