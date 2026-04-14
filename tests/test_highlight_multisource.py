"""Подсветка: несколько источников должны получать разные «владельцы» токенов, где это возможно."""

from highlight import (
    LEVENSHTEIN_LCS_MAX_WORDS,
    _jaccard_multi_token_owners_by_shingle_vote,
    _levenshtein_lcs_word_indices,
    _merge_token_owners_by_pair_scores,
    _tfidf_multi_token_owners_by_feature_vote,
    highlight_jaccard_multi_with_legend,
)


def test_jaccard_vote_gives_distinct_owners_for_segments() -> None:
    raw = "aaa bbb ccc ddd"
    norm = raw.lower()
    n0 = "aaa bbb x"
    n1 = "yyy bbb zzz"
    n2 = "ccc ddd www"
    owners = _jaccard_multi_token_owners_by_shingle_vote(raw, norm, [n0, n1, n2], shingle_k=2)
    # «ccc» и «ddd» сильнее связаны только с третьим соседом
    toks = raw.split()
    idx_ccc = toks.index("ccc")
    idx_ddd = toks.index("ddd")
    assert owners[idx_ccc] == 2
    assert owners[idx_ddd] == 2


def test_jaccard_pair_scores_flip_color_when_shingle_votes_tied() -> None:
    """Одинаковый вклад шинглов — выигрывает сосед с большим % из матрицы (второй цвет в легенде)."""
    raw = "x foo y"
    norm = raw
    hl_first_wins = highlight_jaccard_multi_with_legend(
        raw,
        norm,
        ["x foo a", "x foo b"],
        2,
        ["n0", "n1"],
        neighbor_pair_scores=[90.0, 10.0],
    )
    hl_second_wins = highlight_jaccard_multi_with_legend(
        raw,
        norm,
        ["x foo a", "x foo b"],
        2,
        ["n0", "n1"],
        neighbor_pair_scores=[10.0, 90.0],
    )
    assert "#fecaca" in hl_first_wins.inner_html
    assert "#bfdbfe" in hl_second_wins.inner_html
    assert hl_first_wins.inner_html != hl_second_wins.inner_html


def test_tfidf_vote_can_assign_nonzero_owner_to_second_neighbor() -> None:
    corpus = [
        "unique_one alpha beta",
        "unique_two gamma delta",
        "unique_one unique_two mixed epsilon",
    ]
    raw = corpus[2]
    owners = _tfidf_multi_token_owners_by_feature_vote(raw, corpus, 2, [0, 1], top_n=20)
    toks = raw.split()
    i_u1 = toks.index("unique_one")
    i_u2 = toks.index("unique_two")
    assert owners[i_u1] is not None
    assert owners[i_u2] is not None
    assert owners[i_u1] != owners[i_u2]


def test_levenshtein_lcs_limit_raised() -> None:
    assert LEVENSHTEIN_LCS_MAX_WORDS >= 2500


def test_levenshtein_color_conflict_uses_pair_scores() -> None:
    """Если LCS совпадает у двух соседей, цвет берётся у того, у кого выше % в матрице (как в таблице)."""
    f = "one two three four"
    per = [
        _levenshtein_lcs_word_indices(f, "zero two three nine", 2),
        _levenshtein_lcs_word_indices(f, "alpha two three beta", 2),
    ]
    overlap = per[0] & per[1]
    assert overlap, "нужно пересечение индексов для проверки разрешения конфликта"
    toks = f.split()
    idx_two = toks.index("two")
    assert idx_two in overlap
    owners_lo = _merge_token_owners_by_pair_scores(per, len(toks), [90.0, 10.0])
    owners_hi = _merge_token_owners_by_pair_scores(per, len(toks), [10.0, 90.0])
    assert owners_lo[idx_two] == 0
    assert owners_hi[idx_two] == 1


def test_jaccard_shingle_tie_broken_by_pair_scores() -> None:
    raw = "foo bar baz"
    norm = raw.lower()
    owners = _jaccard_multi_token_owners_by_shingle_vote(
        raw,
        norm,
        ["foo bar x only", "foo bar y only"],
        2,
        neighbor_pair_scores=[5.0, 50.0],
    )
    idx_foo = raw.split().index("foo")
    assert owners[idx_foo] == 1
