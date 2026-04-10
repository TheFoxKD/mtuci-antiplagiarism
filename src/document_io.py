"""Извлечение plain text из поддерживаемых файлов и загрузка корпуса из папки."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from pypdf import PdfReader

from text_normalize import normalize_text

MIN_EXTRACTED_CHARS = 50

SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}


@dataclass
class ExtractOutcome:
    """Результат извлечения текста из одного файла."""

    text: str
    # True если для .txt/.md использовали errors=replace
    encoding_replaced: bool = False


@dataclass
class CorpusEntry:
    """Документ, пригодный для сравнения (после фильтров)."""

    name: str
    path: Path
    raw_text: str
    normalized: str


@dataclass
class LoadCorpusResult:
    """Итог загрузки папки."""

    entries: list[CorpusEntry] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)


def iter_supported_files(directory: Path, *, recursive: bool = False) -> Iterable[Path]:
    """
    Все поддерживаемые файлы в каталоге.
    При recursive=True — обход rglob по подпапкам (глубина без ограничений).
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Не каталог: {directory}")
    paths: list[Path] = []
    if recursive:
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                paths.append(p)
    else:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                paths.append(p)
    return sorted(paths, key=lambda x: str(x.relative_to(directory)).lower())


def extract_plain_text(path: Path) -> ExtractOutcome:
    """Извлекает текст; для короткого PDF-слоя возвращает пустую строку."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        try:
            text = path.read_text(encoding="utf-8")
            return ExtractOutcome(text=text, encoding_replaced=False)
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
            return ExtractOutcome(text=text, encoding_replaced=True)

    if suffix == ".docx":
        from docx import Document as DocxDocument

        doc = DocxDocument(path)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(parts)
        return ExtractOutcome(text=text, encoding_replaced=False)

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        chunks: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                chunks.append(t)
        text = "\n".join(chunks)
        stripped = text.strip()
        if len(stripped) < MIN_EXTRACTED_CHARS:
            return ExtractOutcome(text="", encoding_replaced=False)
        return ExtractOutcome(text=text, encoding_replaced=False)

    raise ValueError(f"Неподдерживаемое расширение: {path}")


def load_corpus(
    directory: Path,
    on_file_error: Callable[[str, str], None] | None = None,
    *,
    recursive: bool = False,
) -> LoadCorpusResult:
    """
    Загружает корпус: пропускает битые файлы, пустые после нормализации, короткий PDF.
    Имя документа — путь относительно directory (важно при recursive=True).
    on_file_error(name, reason) — опциональный колбэк для каждого предупреждения.
    """
    result = LoadCorpusResult()
    for path in iter_supported_files(directory, recursive=recursive):
        name = path.relative_to(directory).as_posix()
        try:
            outcome = extract_plain_text(path)
        except Exception as exc:  # noqa: BLE001 — показываем пользователю причину
            msg = f"ошибка чтения: {exc}"
            result.warnings.append((name, msg))
            if on_file_error:
                on_file_error(name, msg)
            continue

        if outcome.encoding_replaced:
            w = "кодировка UTF-8: недопустимые байты заменены"
            result.warnings.append((name, w))
            if on_file_error:
                on_file_error(name, w)

        raw = outcome.text
        if path.suffix.lower() == ".pdf" and len(raw.strip()) < MIN_EXTRACTED_CHARS:
            msg = (
                "нет извлечённого текста или слишком мало символов "
                f"(<{MIN_EXTRACTED_CHARS}); возможен скан PDF без текстового слоя"
            )
            result.warnings.append((name, msg))
            if on_file_error:
                on_file_error(name, msg)
            continue

        normalized = normalize_text(raw)
        if not normalized:
            msg = "пустой текст после нормализации"
            result.warnings.append((name, msg))
            if on_file_error:
                on_file_error(name, msg)
            continue

        result.entries.append(
            CorpusEntry(name=name, path=path, raw_text=raw, normalized=normalized)
        )

    return result
