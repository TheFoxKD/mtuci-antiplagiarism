"""Три метода сравнения, матрица похожести в процентах, агрегаты по строкам."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_normalize import jaccard_sets, text_to_shingles

# Порог «явно высокой» похожести: такие источники всегда попадают в список, если максимум по строке >= этому уровню.
SIGNIFICANT_SIMILARITY_PCT = 20.0
# Если максимум по строке ниже порога (типично мозаика в Жаккаре/TF‑IDF), в таблицу берём топ‑K соседей по убыванию %.
FALLBACK_TOP_NEIGHBORS_WHEN_WEAK_MAX = 4
# При уже «сильном» максимуме (>= SIGNIFICANT_SIMILARITY_PCT) добавляем соседей близко к нему (ловим несколько источников у смешанного текста).
RELATIVE_BAND_WHEN_STRONG_MAX_PCT = 12.0
MIN_PAIR_PERCENT_FOR_RELATIVE_BAND = 3.0
# Потолок длины списка источников в таблице (защита от шума в больших корпусах).
MAX_NEIGHBORS_LISTED = 8


class SimilarityMethod(str, Enum):
    JACCARD_SHINGLE = "jaccard"
    TFIDF_COSINE = "tfidf"
    LEVENSHTEIN = "levenshtein"


@dataclass
class SimilarityResult:
    """Результат расчёта по корпусу."""

    # симметричная матрица n x n, проценты [0, 100], диагональ 100
    matrix_percent: np.ndarray
    names: list[str]
    # агрегаты по документу
    max_similarity: list[float]
    best_neighbor_index: list[int]
    # Индексы соседей для колонки «источники»: по убыванию %; правило см. _display_neighbor_indices_for_row.
    significant_neighbor_indices: list[list[int]]
    uniqueness_percent: list[float]


def _empty_result(names: list[str]) -> SimilarityResult:
    n = len(names)
    m = np.eye(n) * 100.0 if n else np.zeros((0, 0))
    return SimilarityResult(
        matrix_percent=m,
        names=names,
        max_similarity=[0.0] * n,
        best_neighbor_index=[-1] * n,
        significant_neighbor_indices=[[] for _ in range(n)],
        uniqueness_percent=[100.0] * n,
    )


def _display_neighbor_indices_for_row(
    pairs_sorted_desc: list[tuple[float, int]],
    mval: float,
    *,
    significant_threshold: float = SIGNIFICANT_SIMILARITY_PCT,
) -> list[int]:
    """
    Кого показывать в таблице как «источники» для одной строки матрицы.

    - Если максимум по строке ниже порога (типично кусковой плагиат): топ‑K соседей по %.
    - Если максимум высокий: все с % >= порогу, плюс соседи в полосе (макс − RELATIVE_BAND), с нижним полом.
    """
    if not pairs_sorted_desc:
        return []
    if mval < significant_threshold:
        k = min(FALLBACK_TOP_NEIGHBORS_WHEN_WEAK_MAX, len(pairs_sorted_desc))
        return [j for _, j in pairs_sorted_desc[:k]]
    out: list[int] = []
    seen: set[int] = set()
    band_floor = mval - RELATIVE_BAND_WHEN_STRONG_MAX_PCT
    for s, j in pairs_sorted_desc:
        if len(out) >= MAX_NEIGHBORS_LISTED:
            break
        if s >= significant_threshold or (s >= band_floor and s >= MIN_PAIR_PERCENT_FOR_RELATIVE_BAND):
            if j not in seen:
                seen.add(j)
                out.append(j)
    return out


def _aggregates(
    matrix: np.ndarray,
    significant_threshold: float = SIGNIFICANT_SIMILARITY_PCT,
) -> tuple[list[float], list[int], list[list[int]], list[float]]:
    """
    Для каждой строки: максимум по соседям, лучший индекс, список источников для таблицы,
    оригинальность: 100 − максимум.
    """
    n = matrix.shape[0]
    max_sim: list[float] = []
    best_j: list[int] = []
    significant_all: list[list[int]] = []
    uniq: list[float] = []
    for i in range(n):
        pairs: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            pairs.append((float(matrix[i, j]), j))
        if not pairs:
            max_sim.append(0.0)
            best_j.append(-1)
            significant_all.append([])
            uniq.append(100.0)
            continue
        pairs.sort(key=lambda x: -x[0])
        mval = pairs[0][0]
        bidx = pairs[0][1]
        display_neighbors = _display_neighbor_indices_for_row(
            pairs, mval, significant_threshold=significant_threshold
        )
        max_sim.append(mval)
        best_j.append(bidx)
        significant_all.append(display_neighbors)
        uniq.append(max(0.0, 100.0 - mval))
    return max_sim, best_j, significant_all, uniq


def compute_similarity(
    names: list[str],
    normalized_texts: list[str],
    method: SimilarityMethod,
    shingle_size: int = 2,
    is_cancelled: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SimilarityResult:
    """
    Строит полную матрицу похожести в процентах.
    is_cancelled вызывается между батчами пар.
    progress(done_pairs, total_pairs).
    """
    n = len(names)
    if n == 0:
        return SimilarityResult(
            matrix_percent=np.zeros((0, 0)),
            names=[],
            max_similarity=[],
            best_neighbor_index=[],
            significant_neighbor_indices=[],
            uniqueness_percent=[],
        )

    total_pairs = n * (n - 1) // 2
    done = 0
    batch_every = max(1, total_pairs // 200 or 1)

    def tick() -> None:
        nonlocal done
        done += 1
        if progress and (done % batch_every == 0 or done == total_pairs):
            progress(done, total_pairs)
        if is_cancelled and is_cancelled():
            raise InterruptedError("Расчёт отменён")

    if method == SimilarityMethod.JACCARD_SHINGLE:
        shingles_list = [text_to_shingles(t, shingle_size) for t in normalized_texts]
        matrix = np.eye(n) * 100.0
        for i in range(n):
            for j in range(i + 1, n):
                tick()
                jacc = jaccard_sets(shingles_list[i], shingles_list[j])
                p = 100.0 * jacc
                matrix[i, j] = p
                matrix[j, i] = p
        ms, bj, sg, uq = _aggregates(matrix)
        return SimilarityResult(
            matrix_percent=matrix,
            names=names,
            max_similarity=ms,
            best_neighbor_index=bj,
            significant_neighbor_indices=sg,
            uniqueness_percent=uq,
        )

    if method == SimilarityMethod.TFIDF_COSINE:
        if is_cancelled and is_cancelled():
            raise InterruptedError("Расчёт отменён")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        try:
            x = vectorizer.fit_transform(normalized_texts)
        except ValueError:
            # все пустые — не должно случиться после фильтра корпуса
            return _empty_result(names)
        # Внутри fit_transform отмену не опрашиваем (ограничение sklearn).
        if is_cancelled and is_cancelled():
            raise InterruptedError("Расчёт отменён")
        sim = cosine_similarity(x)
        # обнулить диагональ для отображения пар, потом восстановим 100
        np.fill_diagonal(sim, 0.0)
        matrix = 100.0 * np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(matrix, 100.0)
        ms, bj, sg, uq = _aggregates(matrix)
        return SimilarityResult(
            matrix_percent=matrix,
            names=names,
            max_similarity=ms,
            best_neighbor_index=bj,
            significant_neighbor_indices=sg,
            uniqueness_percent=uq,
        )

    if method == SimilarityMethod.LEVENSHTEIN:
        matrix = np.eye(n) * 100.0
        for i in range(n):
            for j in range(i + 1, n):
                tick()
                a = normalized_texts[i]
                b = normalized_texts[j]
                la, lb = len(a), len(b)
                if la == 0 and lb == 0:
                    p = 100.0
                elif la == 0 or lb == 0:
                    p = 0.0
                else:
                    dist = Levenshtein.distance(a, b)
                    sim = 1.0 - dist / max(la, lb)
                    p = 100.0 * max(0.0, min(1.0, sim))
                matrix[i, j] = p
                matrix[j, i] = p
        ms, bj, sg, uq = _aggregates(matrix)
        return SimilarityResult(
            matrix_percent=matrix,
            names=names,
            max_similarity=ms,
            best_neighbor_index=bj,
            significant_neighbor_indices=sg,
            uniqueness_percent=uq,
        )

    raise ValueError(f"Неизвестный метод: {method}")
