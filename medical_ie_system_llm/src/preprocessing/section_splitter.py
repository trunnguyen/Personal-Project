
from src.models.document import Document
from src.models.section import Section
from src.preprocessing.header_detector import HeaderDetector

class SectionSplitter:

    def process(self, document: Document) -> Document:

        text = document.normalized_text

        self.detector = HeaderDetector()

        matches = self.detector.find_headers(text)

        sections = []

        if not matches:
            # No numbered headers found (e.g. flat, unsectioned free text).
            # Treat the whole document as a single untitled section instead
            # of silently producing zero entities for it.
            sections.append(Section(
                title="",
                text=text,
                start=0,
                end=len(text),
                )
            )

            document.sections = sections

            return document

        # Text before the first header (if any) is otherwise silently
        # dropped and never extracted from.
        first_start = matches[0].start()

        if first_start > 0:
            sections.append(Section(
                title="",
                text=text[:first_start].strip(),
                start=0,
                end=first_start,
                )
            )

        for i, match in enumerate(matches):

            title = match.group(1).strip()

            start = match.start()

            if i == len(matches) - 1:
                end = len(text)
            else:
                end = matches[i + 1].start()

            section_text = text[start:end].strip()

            sections.append(Section(
                title=title,
                text=section_text,
                start=start,
                end=end,
                )
            )

        document.sections = sections

        return document

if __name__ == "__main__":

    from app.config import Config
    from src.utils.file_io import FileLoader
    from src.preprocessing.text_normalizer import DocumentNormalizer

    config = Config()

    loader = FileLoader(config)

    doc = loader.load_all_documents()[0]

    normalizer = DocumentNormalizer()

    doc = normalizer.normalize(doc)

    splitter = SectionSplitter()

    doc = splitter.split(doc)

    for section in doc.sections:

        print("=" *60)

        print(section.title)

        print(section.start, section.end)

        print(section.text[:200])