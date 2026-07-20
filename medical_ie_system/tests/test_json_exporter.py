from app.config import Config

from src.utils.file_io import FileLoader
from src.preprocessing.text_normalizer import DocumentNormalizer
from src.preprocessing.section_splitter import SectionSplitter
from src.extractions.drug_extractor import DrugExtractor
from src.output.json_exporter import JsonExporter

from src.extractions.symptom_extractor import SymptomExtractor

config = Config()

loader = FileLoader(config)

document = loader.load_all_documents()[0]

document = DocumentNormalizer().process(document)
document = SectionSplitter().process(document)

drug_extractor = DrugExtractor()
symptom_extractor = SymptomExtractor()

entities = []

for section in document.sections:
    entities.extend(drug_extractor.extract(section))
    entities.extend(symptom_extractor.extract(section))

JsonExporter().export(
    entities,
    "output.json",
)

print("Done.")