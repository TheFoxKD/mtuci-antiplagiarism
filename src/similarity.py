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

# Левенштейн: при большом произведении длин сравниваем префиксы
LEV_LEN_PRODUCT_THRESHOLD = 2_500_000
LEV_PREFIX_LEN = 4000


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
    # только для метода Левенштейна: True если для пары (i,j) использовали префикс
    levenshtein_prefix_used: np.ndarray | None
    # агрегаты по документу
    max_similarity: list[float]
    best_neighbor_index: list[int]
    uniqueness_percent: list[float]


def _empty_result(names: list[str]) -> SimilarityResult:
    n = len(names)
    m = np.eye(n) * 100.0 if n else np.zeros((0, 0))
    return SimilarityResult(
        matrix_percent=m,
        names=names,
        levenshtein_prefix_used=None,
        max_similarity=[0.0] * n,
        best_neighbor_index=[-1] * n,
        uniqueness_percent=[100.0] * n,
    )


def _aggregates(matrix: np.ndarray) -> tuple[list[float], list[int], list[float]]:
    """max по j!=i, индекс соседа, уникальность = 100 - max."""
    n = matrix.shape[0]
    max_sim: list[float] = []
    best_j: list[int] = []
    uniq: list[float] = []
    for i in range(n):
        best = -1.0
        best_idx = -1
        for j in range(n):
            if i == j:
                continue
            v = matrix[i, j]
            if v > best:
                best = v
                best_idx = j
        max_sim.append(best if best >= 0 else 0.0)
        best_j.append(best_idx)
        uniq.append(100.0 - max_sim[-1])
    return max_sim, best_j, uniq


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
            levenshtein_prefix_used=None,
            max_similarity=[],
            best_neighbor_index=[],
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
        ms, bj, uq = _aggregates(matrix)
        return SimilarityResult(
            matrix_percent=matrix,
            names=names,
            levenshtein_prefix_used=None,
            max_similarity=ms,
            best_neighbor_index=bj,
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
        ms, bj, uq = _aggregates(matrix)
        return SimilarityResult(
            matrix_percent=matrix,
            names=names,
            levenshtein_prefix_used=None,
            max_similarity=ms,
            best_neighbor_index=bj,
            uniqueness_percent=uq,
        )

    if method == SimilarityMethod.LEVENSHTEIN:
        prefix_used = np.zeros((n, n), dtype=bool)
        matrix = np.eye(n) * 100.0
        for i in range(n):
            for j in range(i + 1, n):
                tick()
                a = normalized_texts[i]
                b = normalized_texts[j]
                used_prefix = False
                if len(a) * len(b) > LEV_LEN_PRODUCT_THRESHOLD:
                    a = a[:LEV_PREFIX_LEN]
                    b = b[:LEV_PREFIX_LEN]
                    used_prefix = True
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
                prefix_used[i, j] = used_prefix
                prefix_used[j, i] = used_prefix
        ms, bj, uq = _aggregates(matrix)
        return SimilarityResult(
            matrix_percent=matrix,
            names=names,
            levenshtein_prefix_used=prefix_used,
            max_similarity=ms,
            best_neighbor_index=bj,
            uniqueness_percent=uq,
        )

    raise ValueError(f"Неизвестный метод: {method}")
