import argparse
import re
from itertools import combinations
from pathlib import Path
from typing import Iterable

DEFAULT_INPUT_DIR = Path("data/texts")
DEFAULT_SHINGLE_SIZE = 2
DEFAULT_TOP_RESULTS = 10
TABLE_WIDTH = 72

SPACE_PATTERN = re.compile(r"\s+")
NON_WORD_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)

Document = tuple[str, str]
PreparedDocument = tuple[str, set[str]]
PairSimilarity = tuple[str, str, float]


def parse_args() -> argparse.Namespace:
    """Считывает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Сравнивает текстовые файлы и выводит процент похожести."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Каталог с .txt файлами (по умолчанию: data/texts).",
    )
    parser.add_argument(
        "--shingle-size",
        type=int,
        default=DEFAULT_SHINGLE_SIZE,
        help="Размер шингла в словах (по умолчанию: 2).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_RESULTS,
        help="Сколько лучших пар выводить (по умолчанию: 10).",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Проверяет аргументы запуска."""
    if args.shingle_size < 1:
        raise ValueError("Параметр --shingle-size должен быть >= 1")
    if args.top < 1:
        raise ValueError("Параметр --top должен быть >= 1")


def normalize_text(raw_text: str) -> str:
    """Нормализует текст перед сравнением."""
    text = raw_text.lower()
    text = NON_WORD_PATTERN.sub(" ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def text_to_shingles(text: str, shingle_size: int) -> set[str]:
    """Преобразует текст в множество шинглов."""
    words = text.split()
    if len(words) < shingle_size:
        return {" ".join(words)} if words else set()

    return {
        " ".join(words[index : index + shingle_size])
        for index in range(len(words) - shingle_size + 1)
    }


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Считает коэффициент Жаккара."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def read_txt_files(input_dir: Path) -> list[Document]:
    """Читает все .txt файлы из каталога."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Каталог не найден: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Путь не является каталогом: {input_dir}")

    txt_paths = sorted(input_dir.glob("*.txt"))
    if len(txt_paths) < 2:
        raise ValueError("Для сравнения нужно минимум 2 файла .txt.")

    return [(txt_path.name, txt_path.read_text(encoding="utf-8")) for txt_path in txt_paths]


def prepare_documents(
    documents: Iterable[Document], shingle_size: int
) -> list[PreparedDocument]:
    """Готовит документы к сравнению."""
    prepared: list[PreparedDocument] = []
    for name, raw_text in documents:
        prepared.append((name, text_to_shingles(normalize_text(raw_text), shingle_size)))
    return prepared


def compare_documents(documents: Iterable[Document], shingle_size: int) -> list[PairSimilarity]:
    """Сравнивает все пары документов."""
    prepared_docs = prepare_documents(documents, shingle_size)

    results: list[PairSimilarity] = []
    for (name_a, shingles_a), (name_b, shingles_b) in combinations(prepared_docs, 2):
        results.append((name_a, name_b, jaccard_similarity(shingles_a, shingles_b)))

    results.sort(key=lambda item: item[2], reverse=True)
    return results


def print_results(results: list[PairSimilarity], top: int) -> None:
    """Печатает top результатов."""
    print("Наиболее похожие пары:")
    print("-" * TABLE_WIDTH)
    print(f"{'Файл 1':<25} {'Файл 2':<25} {'Похожесть':>12}")
    print("-" * TABLE_WIDTH)

    for name_a, name_b, score in results[:top]:
        print(f"{name_a:<25} {name_b:<25} {score * 100:>10.2f}%")


def main() -> None:
    """Точка входа."""
    args = parse_args()
    documents = read_txt_files(args.input_dir)
    results = compare_documents(documents, args.shingle_size)
    print_results(results, args.top)


if __name__ == "__main__":
    main()
