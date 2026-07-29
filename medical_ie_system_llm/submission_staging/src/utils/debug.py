def inspect_offset(
    document,
    start=0,
    end=60
):

    print("-" * 80)

    print(
        "NormIdx | OrigIdx | NormChar | OrigChar"
    )

    print("-" * 80)

    mapping = document.offset_map.mapping

    for i in range(start, end):

        orig = mapping[i]

        print(
            f"{i:7} | "
            f"{orig:7} | "
            f"{repr(document.normalized_text[i]):8} | "
            f"{repr(document.text[orig])}"
        )