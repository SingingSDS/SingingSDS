import re
from typing import Optional


def remove_non_zh_jp(text: str) -> str:
    pattern = r"[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f\uff01-\uffef\s]"
    return re.sub(pattern, "", text)


def truncate_sentences(text: str, max_sentences: int) -> str:
    sentences = re.split(r"(?<=[。！？!?~])|(?:\n+)|(?: {2,})", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return "".join(sentences[:max_sentences]).strip()


def clean_llm_output(
    text: str,
    max_sentences: Optional[int] = 2,
    seg_syb: str = " ",
    language: str = "mandarin",
) -> str:
    if language not in ["mandarin", "japanese"]:
        raise NotImplementedError(f"Unsupported language: {language}")
    text = text.strip()
    if max_sentences is not None:
        text = truncate_sentences(text, max_sentences)
    text = remove_non_zh_jp(text)
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = text.replace("\n", seg_syb)
    text = text.replace(" ", seg_syb)
    return text
