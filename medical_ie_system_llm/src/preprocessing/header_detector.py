import re

from typing import Iterable

class HeaderDetector:

    PATTERNS = [
        re.compile(r"^\d+\.\s+(.+)$",
        re.MULTILINE,
                   ),

        # re.compile(r"^([A-ZÀ-Ỹ0-9 ]{5,})$",
        #            re.MULTILINE,
        #            ),
        # re.compile(r"^(.+?):\s*$",
        #            re.MULTILINE,
        #            ),
    ]

    def find_headers(self,text: str,) -> list[re.Match]:

        matches = []

        for pattern in self.PATTERNS:

            matches.extend(
                pattern.finditer(text)
            )

        matches.sort(
            key=lambda m: m.start()
        )

        return matches