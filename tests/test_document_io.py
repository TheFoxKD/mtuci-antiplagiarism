"""Константы загрузки корпуса (ожидания тестировщиков про «лимиты»)."""

from document_io import MIN_EXTRACTED_CHARS


def test_min_extracted_chars_documented() -> None:
    """Файлы короче этого порога отбрасываются как непригодные (см. document_io)."""
    assert MIN_EXTRACTED_CHARS == 50
