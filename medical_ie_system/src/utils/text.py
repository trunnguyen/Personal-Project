import re

def split_sentences(text:str) -> list[str]:

    return[
        s.strip()
        for s in re.split(r"\n+|(?<=[.:;])\s+", text)
        if s.strip()
    ]