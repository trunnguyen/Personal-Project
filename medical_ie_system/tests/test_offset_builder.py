from app.config import Config

from src.utils.file_io import FileLoader

from src.preprocessing.text_normalizer import DocumentNormalizer


def test_all_documents():

    config = Config()

    loader = FileLoader(config)

    docs = loader.load_all_documents()

    normalizer = DocumentNormalizer()

    for doc in docs:

        doc = normalizer.normalize(doc)

        assert len(doc.offset_map.mapping) == len(
            doc.normalized_text
        )

    print("✅ All documents passed offset length test.")
def test_character_mapping():

    config = Config()

    loader = FileLoader(config)

    docs = loader.load_all_documents()

    normalizer = DocumentNormalizer()

    for doc in docs:

        doc = normalizer.normalize(doc)

        mapping = doc.offset_map.mapping

        for i, original_index in enumerate(mapping):

            assert (
                doc.normalized_text[i]
                ==
                doc.text[original_index]
            )

    print("✅ Character mapping verified.")

if __name__ == "__main__":

    test_all_documents()

    test_character_mapping()