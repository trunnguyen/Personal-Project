from app.config import Config
from src.pipeline.medical_ie_pipeline import MedicalIEPipeline

from src.preprocessing.text_normalizer import DocumentNormalizer
from src.preprocessing.section_splitter import SectionSplitter

from src.utils.file_io import FileLoader

config = Config()

loader = FileLoader(config)

pipeline = MedicalIEPipeline(
    stages=[
        DocumentNormalizer()
    ]
)

doc = loader.load_all_documents()[0]

doc = pipeline.process(doc)

print("=" * 60)

for section in doc.sections:
    print(section.title)