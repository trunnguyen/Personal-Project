from app.config import Config

from src.utils.file_io import FileLoader
from src.preprocessing.text_normalizer import DocumentNormalizer
from src.preprocessing.section_splitter import SectionSplitter

from src.extractions.symptom_extractor import SymptomExtractor


config = Config()

loader = FileLoader(config)

document = loader.load_all_documents()[0]

document = DocumentNormalizer().process(document)

document = SectionSplitter().process(document)

extractor = SymptomExtractor()
print("Dictionary size:",
      len(extractor.matcher.dictionary.entries))

for section in document.sections:

    entities = extractor.extract(section)

    for entity in entities:

        print(entity)