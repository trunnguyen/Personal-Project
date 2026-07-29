from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import unicodedata

from src.models.document import Document
from app.config import Config

@dataclass
class FileLoader:
    def __init__(self, config: Config):
        self.config = config
        self.input_dir= Path(self.config.get("paths","input_dir")
                             )
    def load_document(self, file_path: Path) -> Document:
        text = file_path.read_text(
            encoding="utf-8",
        )

        # Confirmed (via direct testing) that these raw files use a MIXED,
        # inconsistent Unicode normalization form (neither purely NFC nor
        # NFD) — not just theoretical risk. This matters because it makes
        # exact-match text comparisons downstream unreliable: the LLM's
        # echoed text (which tokenizers typically produce in NFC) could
        # fail to exact-match a source substring stored in a different
        # normalization form, causing the hallucination guard to wrongly
        # reject a genuinely correct extraction. Normalizing to NFC once,
        # here, at the earliest possible point, makes every downstream
        # comparison (SpanLocator, validate_output.py's keyword checks,
        # offset mapping) operate on a single consistent form. NFC doesn't
        # change the text's visual or semantic content, only how accented
        # characters are internally represented.
        text = unicodedata.normalize("NFC", text)

        return Document(
            doc_id=file_path.stem,
            file_path=file_path,
            text=text,
        )
    def load_all_documents(self) -> List[Document]:

        documents = []

        for file_path in sorted(self.input_dir.glob("*.txt")):
            documents.append(self.load_document(file_path))

        return documents

if __name__ == "__main__":
    config = Config()

    loader = FileLoader(config)

    docs = loader.load_all_documents()

    print(f"Loaded {len(docs)} documents")

    print()

    print("First document:")

    print(docs[0].doc_id)

    print(docs[0].text[:500])