import re

from src.utils.file_io import Document
from src.preprocessing.offset_builder import OffsetBuilder
from src.pipeline.base_stage import PipelineStage

class DocumentNormalizer(PipelineStage):

    def process(self, document: Document) -> Document:

        text = document.text

        # Windows newline
        text = text.replace("\r", "")

        # Collapse repeated spaces
        text = re.sub(r" {2,}"," ", text)
        text = re.sub(r"[ \t]+\n","\n" , text)

        document.normalized_text = text

        builder = OffsetBuilder()
        document.offset_map = builder.build(original = document.text,
                                        normalized = document.normalized_text,)

        return document


if __name__ == "__main__":

    from app.config import Config
    from src.utils.file_io import FileLoader

    config = Config()

    loader = FileLoader(config)

    doc = loader.load_all_documents()[0]

    normalizer = DocumentNormalizer()

    doc = normalizer.normalize(doc)

    print("=" * 60)
    print("NORMALIZED")
    print("=" * 60)

    print(doc.normalized_text[:600])
    #print(repr(doc.normalized_text[450:560]))