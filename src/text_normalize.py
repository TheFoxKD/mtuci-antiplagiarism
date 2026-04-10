"""Единая нормализация текста для всех методов сравнения."""

from __future__ import annotations

import re

SPACE_PATTERN = re.compile(r"\s+")
NON_WORD_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(raw_text: str) -> str:
    """Нижний регистр, убрать лишнее, один пробел между словами."""
    text = raw_text.lower()
    text = NON_WORD_PATTERN.sub(" ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def text_to_shingles(text: str, shingle_size: int) -> set[str]:
    """Множество словесных шинглов."""
    words = text.split()
    if shingle_size < 1:
        raise ValueError("shingle_size >= 1")
    if len(words) < shingle_size:
        return {" ".join(words)} if words else set()
    return {
        " ".join(words[i : i + shingle_size])
        for i in range(len(words) - shingle_size + 1)
    }


def jaccard_sets(left: set[str], right: set[str]) -> float:
    """Коэффициент Жаккара для двух множеств, диапазон [0, 1]."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
