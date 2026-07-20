import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.span_locator import SpanLocator
from src.llm.response_parser import parse_entities, ParseError

# Test 1: hallucination guard rejects invented text
locator = SpanLocator()
section_text = "Bệnh nhân không sốt, không ho."
span = locator.locate(section_text, "đau bụng dữ dội")  # not in text at all
print("Test 1 - hallucinated entity rejected:", span is None)

# Test 2: whitespace-tolerant matching still works
span = locator.locate(section_text, "không   sốt")  # extra spaces vs source
print("Test 2 - whitespace-tolerant match works:", span is not None)

# Test 3: malformed JSON (code fence + trailing commentary) still parses
messy_response = '''Here is the extraction:
```json
[
  {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"], "lookup_term": null}
]
```
Let me know if you need anything else!'''
result = parse_entities(messy_response)
print("Test 3 - parses despite code fence + commentary:", result)

# Test 4: entity with invalid type gets dropped, valid one kept
mixed = '[{"text":"a","type":"BOGUS_TYPE","assertions":[],"lookup_term":null},{"text":"sốt","type":"TRIỆU_CHỨNG","assertions":[],"lookup_term":null}]'
result = parse_entities(mixed)
print("Test 4 - invalid type dropped, valid kept:", len(result) == 1 and result[0]['text']=='sốt')

# Test 5: assertions/candidates stripped for ineligible types
mixed2 = '[{"text":"WBC","type":"TÊN_XÉT_NGHIỆM","assertions":["isNegated"],"lookup_term":"white blood cell"}]'
result = parse_entities(mixed2)
print("Test 5 - lab type has assertions+lookup_term stripped:", result[0]['assertions']==[] and result[0]['lookup_term'] is None)

# Test 6: totally broken JSON doesn't crash the extractor (caught by LlmEntityExtractor)
try:
    parse_entities("I cannot process this request.")
    print("Test 6 - FAILED (should have raised)")
except ParseError:
    print("Test 6 - broken response raises ParseError (caught upstream, section skipped safely)")
