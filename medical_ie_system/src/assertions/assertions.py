from src.models.assertion import Assertion

HISTORICAL_SECTIONS = {
    "tiền sử bệnh",
    "thuốc trước khi nhập viện",
    "thuốc trước nhập viện",
}

class AssertionDetector:

    def detect(self, entity):

        assertions = []

        title = entity.section.title.lower()

        if title in HISTORICAL_SECTIONS:
            assertions.append(Assertion.HISTORICAL)

        return assertions