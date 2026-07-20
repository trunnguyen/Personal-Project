from app.config import Config

from src.pipeline.medical_ie_pipeline import MedicalIEPipeline
from src.utils.file_io import FileLoader

config = Config()

loader = FileLoader(config)

document = loader.load_all_documents()[0]

pipeline = MedicalIEPipeline()

document = pipeline.process(document)

