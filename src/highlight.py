"""Подсветка совпадений для отображения в HTML (QTextEdit)."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from similarity import SimilarityMethod
from text_normalize import normalize_text, text_to_shingles

# Цвета для топ-K соседей (сосед с меньшим индексом в списке = более похожий — приоритет при пересечении).
NEIGHBOR_PALETTE = [
    "#fecaca",
    "#bfdbfe",
    "#bbf7d0",
    "#fde68a",
    "#e9d5ff",
    "#fbcfe8",
    "#a5f3fc",
    "#d9f99d",
    "#fed7aa",
    "#e7e5e4",
]


@dataclass(frozen=True)
class HighlightWithLegend:
    """inner_html — фрагмент для QTextEdit; legend — (имя соседа, #hex)."""

    inner_html: str
    legend: list[tuple[str, str]]


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    """Непустые «слова» и их позиции в исходной строке."""
    spans: list[tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.group(), m.start(), m.end()))
    return spans


def _wrap_html(text: str, highlight_indices: set[int]) -> str:
    """text — исходная строка; highlight_indices — индексы токенов из _token_spans."""
    spans = _token_spans(text)
    if not spans:
        return html.escape(text)

    parts: list[str] = []
    pos = 0
    for i, (tok, start, end) in enumerate(spans):
        if start > pos:
            parts.append(html.escape(text[pos:start]))
        inner = html.escape(tok)
        if i in highlight_indices:
            inner = f'<span style="background-color:#fecaca">{inner}</span>'
        parts.append(inner)
        pos = end
    if pos < len(text):
        parts.append(html.escape(text[pos:]))
    return "".join(parts)


def _merge_neighbor_token_owners(per_neighbor: list[set[int]], n_tokens: int) -> list[int | None]:
    """Для каждого токена — индекс соседа с минимальным номером среди подсветивших (лучший в топ-K)."""
    out: list[int | None] = [None] * n_tokens
    for i in range(n_tokens):
        owners = [j for j, s in enumerate(per_neighbor) if i in s]
        if owners:
            out[i] = min(owners)
    return out


def _wrap_html_by_token_owner(text: str, token_owner: list[int | None]) -> str:
    """Подсветка с разными цветами по индексу соседа (0..K-1)."""
    spans = _token_spans(text)
    if not spans:
        return html.escape(text)
    parts: list[str] = []
    pos = 0
    for i, (tok, start, end) in enumerate(spans):
        if start > pos:
            parts.append(html.escape(text[pos:start]))
        inner = html.escape(tok)
        owner = token_owner[i] if i < len(token_owner) else None
        if owner is not None:
            col = NEIGHBOR_PALETTE[owner % len(NEIGHBOR_PALETTE)]
            inner = f'<span style="background-color:{col}">{inner}</span>'
        parts.append(inner)
        pos = end
    if pos < len(text):
        parts.append(html.escape(text[pos:]))
    return "".join(parts)


def _legend_entries(neighbor_names: list[str]) -> list[tuple[str, str]]:
    return [
        (neighbor_names[j], NEIGHBOR_PALETTE[j % len(NEIGHBOR_PALETTE)])
        for j in range(len(neighbor_names))
    ]


def _wrap_normalized_words_colored(words: list[str], word_owner: list[int | None]) -> str:
    parts: list[str] = []
    for i, w in enumerate(words):
        esc = html.escape(w)
        owner = word_owner[i] if i < len(word_owner) else None
        if owner is not None:
            col = NEIGHBOR_PALETTE[owner % len(NEIGHBOR_PALETTE)]
            esc = f'<span style="background-color:{col}">{esc}</span>'
        parts.append(esc)
    return " ".join(parts)


def _normalized_tokens_from_spans(spans: list[tuple[str, int, int]]) -> list[str]:
    return [normalize_text(t) for t, _, _ in spans]


def _jaccard_highlight_indices(
    raw_f: str, normalized_f: str, normalized_g: str, shingle_k: int
) -> set[int]:
    sf = text_to_shingles(normalized_f, shingle_k)
    sg = text_to_shingles(normalized_g, shingle_k)
    common = sf & sg
    spans = _token_spans(raw_f)
    nt = _normalized_tokens_from_spans(spans)
    highlight: set[int] = set()
    for idx in range(len(nt)):
        for sh in common:
            for w in sh.split():
                if w == nt[idx]:
                    highlight.add(idx)
                    break
    return highlight


def highlight_jaccard(raw_f: str, normalized_f: str, normalized_g: str, shingle_k: int) -> str:
    """Подсветка слов F, входящих в общие шинглы с G."""
    h = _jaccard_highlight_indices(raw_f, normalized_f, normalized_g, shingle_k)
    return _wrap_html(raw_f, h)


def highlight_jaccard_multi(
    raw_f: str, normalized_f: str, neighbors_norm: list[str], shingle_k: int
) -> str:
    """Объединение подсветки по нескольким соседям (один цвет)."""
    merged: set[int] = set()
    for ng in neighbors_norm:
        merged |= _jaccard_highlight_indices(raw_f, normalized_f, ng, shingle_k)
    return _wrap_html(raw_f, merged)


def _tfidf_highlight_indices(
    raw_f: str,
    corpus_normalized: list[str],
    index_f: int,
    index_g: int,
    top_n: int,
) -> set[int]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x = vec.fit_transform(corpus_normalized)
    row_f = x[index_f].toarray().ravel()
    row_g = x[index_g].toarray().ravel()
    names = vec.get_feature_names_out()
    order = np.argsort(row_f)[::-1]
    keywords: set[str] = set()
    for idx in order[:top_n]:
        if row_f[idx] <= 0:
            continue
        if row_g[idx] <= 0:
            continue
        for part in names[idx].split():
            keywords.add(normalize_text(part))
    spans = _token_spans(raw_f)
    nt = _normalized_tokens_from_spans(spans)
    return {i for i, t in enumerate(nt) if t in keywords}


def highlight_tfidf(
    raw_f: str,
    corpus_normalized: list[str],
    index_f: int,
    index_g: int,
    top_n: int = 30,
) -> str:
    """
    Эвристика: топ-N признаков по весу TF-IDF в F, пересекающиеся с ненулевым весом в G.
    Подсвечиваем токены F, совпадающие с этими признаками (слова и части биграмм).
    """
    h = _tfidf_highlight_indices(raw_f, corpus_normalized, index_f, index_g, top_n)
    return _wrap_html(raw_f, h)


def highlight_tfidf_multi(
    raw_f: str,
    corpus_normalized: list[str],
    index_f: int,
    neighbor_indices: list[int],
    top_n: int = 30,
) -> str:
    merged: set[int] = set()
    for j in neighbor_indices:
        merged |= _tfidf_highlight_indices(raw_f, corpus_normalized, index_f, j, top_n)
    return _wrap_html(raw_f, merged)


def _lcs_word_indices(a: list[str], b: list[str]) -> set[int]:
    """Индексы в a, входящие в LCS (для подсветки)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return set()
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # backtrack
    i, j = na, nb
    use_i: set[int] = set()
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            use_i.add(i - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return use_i


def _levenshtein_lcs_word_indices(normalized_f: str, normalized_g: str, min_run: int) -> set[int]:
    wf = normalized_f.split()
    wg = normalized_g.split()
    if len(wf) > 800 or len(wg) > 800:
        wf = wf[:800]
        wg = wg[:800]
    lcs_i = _lcs_word_indices(wf, wg)
    highlight: set[int] = set()
    current: list[int] = []
    for i in range(len(wf)):
        if i in lcs_i:
            current.append(i)
        else:
            if len(current) >= min_run:
                highlight.update(current)
            current = []
    if len(current) >= min_run:
        highlight.update(current)
    return highlight


def highlight_levenshtein_lcs_normalized(normalized_f: str, normalized_g: str, min_run: int = 2) -> str:
    """
    LCS по словам нормализованного текста; подсвечиваем только цепочки >= min_run.
    Показываем нормализованный текст (совпадает с пайплайном метрики).
    """
    wf = normalized_f.split()
    highlight = _levenshtein_lcs_word_indices(normalized_f, normalized_g, min_run)
    parts: list[str] = []
    for i, w in enumerate(wf):
        esc = html.escape(w)
        if i in highlight:
            esc = f'<span style="background-color:#fecaca">{esc}</span>'
        parts.append(esc)
    return " ".join(parts)


def highlight_levenshtein_multi_normalized(
    normalized_f: str, neighbors_norm: list[str], min_run: int = 2
) -> str:
    """Объединение подсветки LCS по нескольким соседям."""
    wf = normalized_f.split()
    merged: set[int] = set()
    for ng in neighbors_norm:
        merged |= _levenshtein_lcs_word_indices(normalized_f, ng, min_run)
    parts: list[str] = []
    for i, w in enumerate(wf):
        esc = html.escape(w)
        if i in merged:
            esc = f'<span style="background-color:#fecaca">{esc}</span>'
        parts.append(esc)
    return " ".join(parts)


def highlight_jaccard_multi_with_legend(
    raw_f: str,
    normalized_f: str,
    neighbors_norm: list[str],
    shingle_k: int,
    neighbor_names: list[str],
) -> HighlightWithLegend:
    """Топ-K соседей: разные цвета; при пересечении токена — цвет более похожего соседа."""
    per = [_jaccard_highlight_indices(raw_f, normalized_f, ng, shingle_k) for ng in neighbors_norm]
    n_tok = len(_token_spans(raw_f))
    owners = _merge_neighbor_token_owners(per, n_tok)
    inner = _wrap_html_by_token_owner(raw_f, owners)
    return HighlightWithLegend(inner_html=inner, legend=_legend_entries(neighbor_names))


def highlight_tfidf_multi_with_legend(
    raw_f: str,
    corpus_normalized: list[str],
    index_f: int,
    neighbor_indices: list[int],
    neighbor_names: list[str],
    top_n: int = 30,
) -> HighlightWithLegend:
    per = [
        _tfidf_highlight_indices(raw_f, corpus_normalized, index_f, j, top_n)
        for j in neighbor_indices
    ]
    n_tok = len(_token_spans(raw_f))
    owners = _merge_neighbor_token_owners(per, n_tok)
    inner = _wrap_html_by_token_owner(raw_f, owners)
    return HighlightWithLegend(inner_html=inner, legend=_legend_entries(neighbor_names))


def highlight_levenshtein_multi_with_legend(
    normalized_f: str,
    neighbors_norm: list[str],
    neighbor_names: list[str],
    min_run: int = 2,
) -> HighlightWithLegend:
    wf = normalized_f.split()
    per = [_levenshtein_lcs_word_indices(normalized_f, ng, min_run) for ng in neighbors_norm]
    owners = _merge_neighbor_token_owners(per, len(wf))
    inner = _wrap_normalized_words_colored(wf, owners)
    return HighlightWithLegend(inner_html=inner, legend=_legend_entries(neighbor_names))


def build_highlight_html(
    method: SimilarityMethod,
    raw_f: str,
    raw_g: str,
    normalized_f: str,
    normalized_g: str,
    *,
    shingle_k: int = 2,
    tfidf_corpus: list[str] | None = None,
    tfidf_index_f: int = 0,
    tfidf_index_g: int = 1,
    tfidf_top_n: int = 30,
) -> str:
    if method == SimilarityMethod.JACCARD_SHINGLE:
        return highlight_jaccard(raw_f, normalized_f, normalized_g, shingle_k)
    if method == SimilarityMethod.TFIDF_COSINE:
        if tfidf_corpus is None:
            return html.escape(raw_f)
        return highlight_tfidf(raw_f, tfidf_corpus, tfidf_index_f, tfidf_index_g, tfidf_top_n)
    if method == SimilarityMethod.LEVENSHTEIN:
        return highlight_levenshtein_lcs_normalized(normalized_f, normalized_g)
    return html.escape(raw_f)


def html_document_body(inner_html: str) -> str:
    return (
        "<html><body style=\"font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;"
        "line-height:1.45;color:#1e293b\">"
        f"{inner_html}</body></html>"
    )
