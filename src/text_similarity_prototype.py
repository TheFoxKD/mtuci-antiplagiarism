"""CLI: сравнение документов в папке (все поддерживаемые форматы)."""

from __future__ import annotations

import argparse
from pathlib import Path

from document_io import load_corpus
from similarity import SimilarityMethod, compute_similarity

DEFAULT_INPUT_DIR = Path("data/texts")
DEFAULT_SHINGLE_SIZE = 2
DEFAULT_TOP_RESULTS = 10
TABLE_WIDTH = 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнивает документы в папке и выводит топ пар по похожести (шинглы + Жаккар)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Каталог с .txt, .md, .docx, .pdf",
    )
    parser.add_argument(
        "--shingle-size",
        type=int,
        default=DEFAULT_SHINGLE_SIZE,
        help="Размер шинга (по умолчанию: 2).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_RESULTS,
        help="Сколько лучших пар выводить.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Искать файлы во вложенных папках (имена — относительный путь).",
    )
    args = parser.parse_args()
    if args.shingle_size < 1:
        raise ValueError("Параметр --shingle-size должен быть >= 1")
    if args.top < 1:
        raise ValueError("Параметр --top должен быть >= 1")
    return args


def _pairs_from_matrix(
    names: list[str], matrix_percent, top: int
) -> list[tuple[str, str, float]]:
    """Пары i<j с похожестью в долях [0,1] для вывода как проценты."""
    n = len(names)
    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((names[i], names[j], matrix_percent[i, j] / 100.0))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top]


def main() -> None:
    args = parse_args()
    loaded = load_corpus(args.input_dir, recursive=args.recursive)
    if loaded.warnings:
        for name, msg in loaded.warnings:
            print(f"[предупреждение] {name}: {msg}")
    if len(loaded.entries) < 2:
        print("Ошибка: нужно минимум 2 пригодных документа.")
        raise SystemExit(1)
    names = [e.name for e in loaded.entries]
    texts = [e.normalized for e in loaded.entries]
    res = compute_similarity(
        names,
        texts,
        SimilarityMethod.JACCARD_SHINGLE,
        shingle_size=args.shingle_size,
    )
    pairs = _pairs_from_matrix(names, res.matrix_percent, args.top)
    print("Наиболее похожие пары:")
    print("-" * TABLE_WIDTH)
    print(f"{'Файл 1':<25} {'Файл 2':<25} {'Похожесть':>12}")
    print("-" * TABLE_WIDTH)
    for name_a, name_b, score in pairs:
        print(f"{name_a:<25} {name_b:<25} {score * 100:>10.2f}%")


if __name__ == "__main__":
    main()
