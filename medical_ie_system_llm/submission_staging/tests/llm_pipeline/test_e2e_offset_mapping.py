import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.config import Config
from src.utils.file_io import FileLoader
from src.preprocessing.text_normalizer import DocumentNormalizer
from src.preprocessing.section_splitter import SectionSplitter
from src.extractions.llm_extractor import LlmEntityExtractor
from src.extractions.llm_medical_entity_extractor import LlmMedicalEntityExtractor
from src.llm.client import FakeLLMClient
from src.retrieval.fuzzy_code_retriever import FuzzyCodeRetriever, format_icd10_code
from src.output.json_exporter import JsonExporter

config = Config()
loader = FileLoader(config)
doc = next(d for d in loader.load_all_documents() if d.doc_id == '1')
doc = DocumentNormalizer().process(doc)
doc = SectionSplitter().process(doc)

section1 = doc.sections[0]
print("SECTION 1 TEXT (normalized):")
print(repr(section1.text))
print()

# Build a scripted response matching what a good LLM extraction would look
# like for this real section, including the entity that sits right after
# the triple-space run that normalization collapses ("Căng thẳng").
scripted_response = '''[
  {"text": "metoprolol 25mg po bid", "type": "THUỐC", "assertions": ["isHistorical"], "lookup_term": "metoprolol 25 mg oral tablet"},
  {"text": "doxycycline", "type": "THUỐC", "assertions": ["isHistorical"], "lookup_term": "doxycycline 100 mg oral tablet"},
  {"text": "atenolol", "type": "THUỐC", "assertions": ["isHistorical"], "lookup_term": "atenolol oral tablet"},
  {"text": "Căng thẳng", "type": "TRIỆU_CHỨNG", "assertions": ["isHistorical"], "lookup_term": null},
  {"text": "caffeine", "type": "THUỐC", "assertions": ["isHistorical"], "lookup_term": "caffeine oral"}
]'''

fake_client = FakeLLMClient(scripted_responses={section1.text[:40]: scripted_response})

icd_retriever = FuzzyCodeRetriever('data/knowledge/icd10.csv', code_col='code', name_col='name', code_formatter=format_icd10_code)
rx_retriever = FuzzyCodeRetriever('data/knowledge/drugs.csv', code_col='concept_id', name_col='name')

llm_extractor = LlmEntityExtractor(fake_client, icd_retriever, rx_retriever)
orchestrator = LlmMedicalEntityExtractor(llm_extractor)

doc = orchestrator.process(doc)

print(f"Extracted {len(doc.entities)} entities\n")

all_correct = True
for e in doc.entities:
    # THE critical round-trip check: does slicing the ORIGINAL raw text at
    # [start:end] give back exactly the entity's text? If offset_map wasn't
    # applied correctly, this will fail for "Căng thẳng" (comes after the
    # collapsed triple-space).
    actual_slice = doc.text[e.start:e.end]
    correct = actual_slice == e.text
    all_correct &= correct
    status = "OK" if correct else "MISMATCH"
    print(f"[{status}] {e.entity_type.value:20s} text={e.text!r:35s} pos=({e.start},{e.end}) original_slice={actual_slice!r} candidates={e.candidates}")

print()
print("ALL POSITIONS CORRECT AGAINST ORIGINAL RAW FILE:" , all_correct)

# Also test the exporter's conditional field behavior
import json
JsonExporter().export(doc.entities, '/tmp/test_output.json')
with open('/tmp/test_output.json', encoding='utf-8') as f:
    exported = json.load(f)
print()
print("Exported JSON sample (first entity):")
print(json.dumps(exported[0], ensure_ascii=False, indent=2))
print()
print("Symptom entity (should have assertions but NO candidates key):")
symptom_entry = next(x for x in exported if x['type'] == 'TRIỆU_CHỨNG')
print(json.dumps(symptom_entry, ensure_ascii=False, indent=2))
print("'candidates' key absent for symptom:", 'candidates' not in symptom_entry)
