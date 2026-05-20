import re

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MENTION_PATTERN = re.compile(r"@([a-zA-Z][a-zA-Z0-9_]{3,30}[a-zA-Z0-9])")


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.replace("\x00", "")
    text = _ZERO_WIDTH.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def extract_mentions(text: str | None) -> list[str]:
    if not text:
        return []
    return _MENTION_PATTERN.findall(text)
