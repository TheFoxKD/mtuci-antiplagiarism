"""Правила списка источников в таблице (агрегаты по строке матрицы)."""

from similarity import (
    FALLBACK_TOP_NEIGHBORS_WHEN_WEAK_MAX,
    SIGNIFICANT_SIMILARITY_PCT,
    _display_neighbor_indices_for_row,
    compute_similarity,
    SimilarityMethod,
)


def test_weak_max_uses_top_k_neighbors() -> None:
    pairs = [(8.7, 0), (7.5, 1), (6.2, 2), (5.1, 3), (4.0, 4)]
    out = _display_neighbor_indices_for_row(pairs, mval=8.7)
    assert out == [0, 1, 2, 3]
    assert len(out) == FALLBACK_TOP_NEIGHBORS_WHEN_WEAK_MAX


def test_strong_max_includes_relative_band() -> None:
    pairs = [(26.65, 0), (21.21, 1), (18.81, 2), (5.0, 3)]
    out = _display_neighbor_indices_for_row(pairs, mval=26.65)
    assert out == [0, 1, 2]
    assert 3 not in out


def test_uniqueness_is_100_minus_max_only() -> None:
    names = ["a", "b", "c"]
    texts = ["alpha beta gamma", "alpha beta delta", "epsilon zeta eta"]
    r = compute_similarity(names, texts, SimilarityMethod.JACCARD_SHINGLE, shingle_size=2)
    for i in range(3):
        assert abs(r.uniqueness_percent[i] - (100.0 - r.max_similarity[i])) < 1e-9


def test_no_significant_threshold_confusion_row() -> None:
    """При низком максимуме всё равно есть список соседей (не пустой «нет совпадений» из-за 20%)."""
    names = list("abcde")
    # Пары не пересекаются по шинглам — максимум ~0, срабатывает топ‑K fallback.
    texts = [" ".join([f"w{i}_{k}" for k in range(50)]) for i in range(5)]
    r = compute_similarity(names, texts, SimilarityMethod.JACCARD_SHINGLE, shingle_size=2)
    assert r.max_similarity[0] < SIGNIFICANT_SIMILARITY_PCT
    assert len(r.significant_neighbor_indices[0]) == FALLBACK_TOP_NEIGHBORS_WHEN_WEAK_MAX
