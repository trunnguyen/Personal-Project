from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from app.config import Config

@dataclass
class Document:
    doc_id: str
    file_path: Path
    text: str

    normalized_text: str =""

    offset_map: object = None
    sections: list = field(default_factory=list)

    entities: list = field(default_factory=list)
class FileLoader:
    def __init__(self, config: Config):
        self.config = config
        self.input_dir= Path(self.config.get("paths","input_dir")
                             )
    def load_document(self, file_path: Path) -> Document:
        text = file_path.read_text(
            encoding="utf-8",
        )

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