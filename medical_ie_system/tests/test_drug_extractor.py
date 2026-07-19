from app.config import Config

from src.preprocessing.text_normalizer import DocumentNormalizer
from src.preprocessing.section_splitter import SectionSplitter
from src.extractions.drug_extractor import DrugExtractor
from src.utils.file_io import FileLoader


config = Config()

loader = FileLoader(config)

document = loader.load_all_documents()[0]

document = DocumentNormalizer().process(document)

document = SectionSplitter().process(document)

extractor = DrugExtractor()

for section in document.sections:

    entities = extractor.extract(section)

    for entity in entities:

        print(entity.text)