from src.models.offset_map import OffsetMap


class OffsetBuilder:

    def build(
        self,
        original: str,
        normalized: str,
    ) -> OffsetMap:

        mapping = []

        i = 0      # original pointer
        j = 0      # normalized pointer

        while j < len(normalized):

            # Original ended before normalized
            if i >= len(original):
                raise ValueError(
                    "Normalized text is longer than original text."
                )

            if original[i] == normalized[j]:
                mapping.append(i)
                i += 1
                j += 1

            else:
                # Skip removed character
                i += 1

        offset_map = OffsetMap(mapping)

        self.validate(
            original,
            normalized,
            offset_map
        )

        return offset_map


    def validate(
        self,
        original: str,
        normalized: str,
        offset_map: OffsetMap
    ) -> None:

        if len(offset_map.mapping) != len(normalized):

            raise ValueError(
                "Offset mapping length does not match normalized text."
            )

        for i, original_index in enumerate(offset_map.mapping):

            if normalized[i] != original[original_index]:

                raise ValueError(
                    f"""
Offset mapping failed.

Normalized index : {i}
Original index   : {original_index}

Normalized char  : {repr(normalized[i])}
Original char    : {repr(original[original_index])}
"""
                )


if __name__ == "__main__":

    original = "Không  ho"

    normalized = "Không ho"

    builder = OffsetBuilder()

    offset = builder.build(
        original,
        normalized
    )

    print(offset.mapping)