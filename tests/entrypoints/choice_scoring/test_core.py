# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-logic unit tests for choice scoring & ranking.

These never touch a model or GPU: a deterministic in-memory ``score_fn`` lets
us pin down every aggregation, selection, tie-break, autoregressive-conditioning
and edge-case rule. The integration tests (offline ``LLM`` + HTTP) live
alongside and exercise the same logic end-to-end against a real model.
"""

from __future__ import annotations

import pytest

from vllm.entrypoints.choice_scoring.core import (
    aggregate_choice,
    rank_select,
    score_choices,
    select_best_index,
)
from vllm.entrypoints.choice_scoring.params import Candidate


# --------------------------------------------------------------------------- #
# Test doubles for score_fn
# --------------------------------------------------------------------------- #
def fixed_scorer(table, ranks_table=None, record=None):
    """A score_fn that ignores context and looks each candidate up in ``table``.

    ``table`` maps tuple(token_ids) -> list[float] (per-token logprobs).
    ``ranks_table`` optionally maps tuple(token_ids) -> list[int].
    ``record`` optional list to append each (context, [cands]) call to.
    """

    def _fn(context, cands):
        if record is not None:
            record.append((list(context), [list(c) for c in cands]))
        out = []
        for c in cands:
            key = tuple(c)
            ranks = None if ranks_table is None else ranks_table[key]
            out.append((list(table[key]), ranks))
        return out

    return _fn


def context_scorer(score_of, record=None):
    """A context-*dependent* score_fn, to verify autoregressive re-scoring.

    ``score_of(last_context_token, cand_token_ids) -> list[float]``.
    """

    def _fn(context, cands):
        if record is not None:
            record.append((list(context), [list(c) for c in cands]))
        last = context[-1] if context else None
        return [(list(score_of(last, c)), None) for c in cands]

    return _fn


def cand(token_ids, index=0, text=None, tokens=None):
    return Candidate(token_ids=token_ids, index=index, text=text, tokens=tokens)


# --------------------------------------------------------------------------- #
# aggregate_choice
# --------------------------------------------------------------------------- #
def test_aggregate_sum_and_mean():
    cs = aggregate_choice(cand([1, 2, 3]), [-1.0, -2.0, -3.0])
    assert cs.sum_logprob == pytest.approx(-6.0)
    assert cs.mean_logprob == pytest.approx(-2.0)
    assert cs.token_logprobs == [-1.0, -2.0, -3.0]
    assert cs.token_ids == [1, 2, 3]


def test_aggregate_single_token_mean_equals_sum():
    cs = aggregate_choice(cand([7]), [-0.5])
    assert cs.sum_logprob == pytest.approx(-0.5)
    assert cs.mean_logprob == pytest.approx(-0.5)


def test_aggregate_is_greedy_true_when_all_rank_one():
    cs = aggregate_choice(cand([1, 2]), [-0.1, -0.2], ranks=[1, 1])
    assert cs.is_greedy is True


def test_aggregate_is_greedy_false_when_any_rank_above_one():
    cs = aggregate_choice(cand([1, 2]), [-0.1, -0.2], ranks=[1, 4])
    assert cs.is_greedy is False


def test_aggregate_is_greedy_none_without_ranks():
    cs = aggregate_choice(cand([1, 2]), [-0.1, -0.2])
    assert cs.is_greedy is None


def test_aggregate_carries_text_and_tokens():
    cs = aggregate_choice(
        cand([1, 2], text=" Paris", tokens=[" Par", "is"]), [-0.1, -0.2]
    )
    assert cs.text == " Paris"
    assert cs.tokens == [" Par", "is"]


def test_aggregate_empty_tokens_raises():
    with pytest.raises(ValueError, match="zero tokens"):
        aggregate_choice(cand([]), [])


def test_aggregate_logprob_length_mismatch_raises():
    with pytest.raises(ValueError, match="logprobs"):
        aggregate_choice(cand([1, 2]), [-0.1])


def test_aggregate_rank_length_mismatch_raises():
    with pytest.raises(ValueError, match="ranks"):
        aggregate_choice(cand([1, 2]), [-0.1, -0.2], ranks=[1])


# --------------------------------------------------------------------------- #
# select_best_index
# --------------------------------------------------------------------------- #
def _score(index, token_ids, logprobs):
    return aggregate_choice(cand(token_ids, index=index), logprobs)


def test_select_best_by_mean():
    choices = [
        _score(0, [1, 2], [-1.0, -1.0]),  # mean -1.0
        _score(1, [3], [-0.5]),  # mean -0.5  <-- best by mean
    ]
    assert select_best_index(choices, "mean") == 1


def test_select_best_by_sum_diverges_from_mean():
    # Different lengths make sum and mean disagree.
    choices = [
        _score(0, [1, 2], [-1.0, -1.0]),  # sum -2.0, mean -1.0
        _score(1, [3], [-1.5]),  # sum -1.5, mean -1.5
    ]
    assert select_best_index(choices, "sum") == 1  # -1.5 > -2.0
    assert select_best_index(choices, "mean") == 0  # -1.0 > -1.5


def test_select_best_tie_breaks_to_lowest_index():
    choices = [
        _score(0, [1], [-1.0]),
        _score(1, [2], [-1.0]),  # exact tie
    ]
    assert select_best_index(choices, "mean") == 0


def test_select_best_empty_returns_minus_one():
    assert select_best_index([], "mean") == -1


def test_select_best_unknown_metric_raises():
    with pytest.raises(ValueError, match="select_by"):
        select_best_index([_score(0, [1], [-1.0])], "median")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# score_choices
# --------------------------------------------------------------------------- #
def test_score_choices_basic():
    table = {(10,): [-0.5], (20, 21): [-1.0, -2.0]}
    out = score_choices(
        [1, 2, 3],
        [cand([10], index=0), cand([20, 21], index=1)],
        fixed_scorer(table),
    )
    assert out.prompt_token_ids == [1, 2, 3]
    assert len(out.choices) == 2
    assert out.choices[0].sum_logprob == pytest.approx(-0.5)
    assert out.choices[1].sum_logprob == pytest.approx(-3.0)
    # mean: -0.5 vs -1.5 -> choice 0 best
    assert out.best_choice_index == 0


def test_score_choices_different_lengths_no_uniform_assumption():
    table = {(10,): [-0.4], (20, 21, 22): [-0.1, -0.1, -0.1]}
    out = score_choices(
        [1],
        [cand([10], index=0), cand([20, 21, 22], index=1)],
        fixed_scorer(table),
    )
    # mean: -0.4 vs -0.1 -> choice 1 best
    assert out.best_choice_index == 1


def test_score_choices_select_by_sum_changes_winner():
    # choice0 (len 2): sum -0.2, mean -0.1  -> best by mean
    # choice1 (len 1): sum -0.15, mean -0.15 -> best by sum
    table = {(10, 11): [-0.1, -0.1], (20,): [-0.15]}
    cands = [cand([10, 11], index=0), cand([20], index=1)]
    by_mean = score_choices([1], cands, fixed_scorer(table), select_by="mean")
    by_sum = score_choices([1], cands, fixed_scorer(table), select_by="sum")
    assert by_mean.best_choice_index == 0  # mean -0.1 > -0.15
    assert by_sum.best_choice_index == 1  # sum -0.15 > -0.2


def test_score_choices_passes_prompt_as_context():
    record = []
    table = {(10,): [-0.5]}
    score_choices([9, 8, 7], [cand([10], index=0)], fixed_scorer(table, record=record))
    assert record[0][0] == [9, 8, 7]


def test_score_choices_carries_metadata():
    table = {(10,): [-0.5]}
    out = score_choices(
        [1],
        [cand([10], index=0, text=" yes", tokens=[" yes"])],
        fixed_scorer(table),
    )
    assert out.choices[0].text == " yes"
    assert out.choices[0].tokens == [" yes"]


def test_score_choices_is_greedy_propagates():
    table = {(10,): [-0.5], (20,): [-0.6]}
    ranks = {(10,): [1], (20,): [3]}
    out = score_choices(
        [1],
        [cand([10], index=0), cand([20], index=1)],
        fixed_scorer(table, ranks_table=ranks),
    )
    assert out.choices[0].is_greedy is True
    assert out.choices[1].is_greedy is False


def test_score_choices_empty_pool_raises():
    with pytest.raises(ValueError, match="at least one"):
        score_choices([1], [], fixed_scorer({}))


def test_score_choices_empty_choice_token_raises():
    with pytest.raises(ValueError, match="zero tokens"):
        score_choices([1], [cand([], index=0)], fixed_scorer({}))


def test_score_choices_scorer_wrong_length_raises():
    def bad_fn(context, cands):
        return []  # too few

    with pytest.raises(ValueError, match="returned 0 results"):
        score_choices([1], [cand([10], index=0)], bad_fn)


# --------------------------------------------------------------------------- #
# rank_select
# --------------------------------------------------------------------------- #
def _three_cands():
    return [cand([10], index=0), cand([11], index=1), cand([12], index=2)]


def test_rank_k1_returns_single_best():
    table = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0]}
    out = rank_select([1], _three_cands(), k=1, score_fn=fixed_scorer(table))
    assert out.truncated is False
    assert len(out.selected) == 1
    assert out.selected[0].choice_index == 1  # -1.0 best
    assert out.selected[0].order == 0


def test_rank_k_equals_pool_is_full_ordering():
    table = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0]}
    out = rank_select([1], _three_cands(), k=3, score_fn=fixed_scorer(table))
    assert out.truncated is False
    assert [s.choice_index for s in out.selected] == [1, 2, 0]
    assert [s.order for s in out.selected] == [0, 1, 2]


def test_rank_k_greater_than_pool_truncates():
    table = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0]}
    out = rank_select([1], _three_cands(), k=10, score_fn=fixed_scorer(table))
    assert out.truncated is True
    assert len(out.selected) == 3  # exhausted, not 10
    assert [s.choice_index for s in out.selected] == [1, 2, 0]


def test_rank_k_zero_returns_empty():
    table = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0]}
    out = rank_select([1], _three_cands(), k=0, score_fn=fixed_scorer(table))
    assert out.selected == []
    assert out.truncated is False


def test_rank_negative_k_raises():
    with pytest.raises(ValueError, match="k must be >= 0"):
        rank_select([1], _three_cands(), k=-1, score_fn=fixed_scorer({}))


def test_rank_empty_pool_raises():
    with pytest.raises(ValueError, match="at least one"):
        rank_select([1], [], k=1, score_fn=fixed_scorer({}))


def test_rank_no_duplicate_selections():
    table = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0]}
    out = rank_select([1], _three_cands(), k=3, score_fn=fixed_scorer(table))
    picked = [s.choice_index for s in out.selected]
    assert len(picked) == len(set(picked))


def test_rank_tie_breaks_to_lowest_index():
    table = {(10,): [-1.0], (11,): [-1.0], (12,): [-1.0]}  # all tied
    out = rank_select([1], _three_cands(), k=3, score_fn=fixed_scorer(table))
    assert [s.choice_index for s in out.selected] == [0, 1, 2]


def test_rank_is_autoregressive_not_static_sort():
    # Initial (context ends in 0): preference A(10) > B(11) > C(12).
    # After A (context ends in 10): C(12) > B(11).
    # After C (context ends in 12): B(11).
    # => order A, C, B == [0, 2, 1]. A *static* sort by initial scores would
    # give [0, 1, 2]; proving re-scoring against the growing context.
    def score_of(last, c):
        first = c[0]
        if last == 0:
            return [{10: -1.0, 11: -2.0, 12: -3.0}[first]]
        if last == 10:
            return [{11: -5.0, 12: -0.5}[first]]
        if last == 12:
            return [{11: -0.1}[first]]
        raise AssertionError(f"unexpected context tail {last}")

    out = rank_select([0], _three_cands(), k=3, score_fn=context_scorer(score_of))
    assert [s.choice_index for s in out.selected] == [0, 2, 1]


def test_rank_context_grows_by_chosen_bundle_each_step():
    record = []
    table = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0]}
    rank_select(
        [1, 2], _three_cands(), k=3, score_fn=fixed_scorer(table, record=record)
    )
    # Step 0 context = prompt. Step 1 += best([11]). Step 2 += next best([12]).
    assert record[0][0] == [1, 2]
    assert record[1][0] == [1, 2, 11]
    assert record[2][0] == [1, 2, 11, 12]
    # Pool shrinks: 3 -> 2 -> 1 candidates re-scored.
    assert [len(call[1]) for call in record] == [3, 2, 1]


def test_rank_multitoken_bundles_use_mean_for_selection():
    # A=[10,11] mean -1.0 ; B=[12] mean -1.5  -> A wins by mean.
    table = {(10, 11): [-1.0, -1.0], (12,): [-1.5]}
    cands = [cand([10, 11], index=0), cand([12], index=1)]
    out = rank_select([1], cands, k=1, score_fn=fixed_scorer(table), select_by="mean")
    assert out.selected[0].choice_index == 0
    assert out.selected[0].mean_logprob == pytest.approx(-1.0)
    assert out.selected[0].sum_logprob == pytest.approx(-2.0)


def test_rank_select_by_sum_prefers_longer_high_total():
    # A=[10,11] sum -2.0 ; B=[12] sum -1.5 -> B wins by sum.
    table = {(10, 11): [-1.0, -1.0], (12,): [-1.5]}
    cands = [cand([10, 11], index=0), cand([12], index=1)]
    out = rank_select([1], cands, k=1, score_fn=fixed_scorer(table), select_by="sum")
    assert out.selected[0].choice_index == 1


def test_rank_records_per_token_logprobs():
    table = {(10, 11): [-0.3, -0.7], (12,): [-5.0]}
    cands = [cand([10, 11], index=0), cand([12], index=1)]
    out = rank_select([1], cands, k=1, score_fn=fixed_scorer(table))
    assert out.selected[0].token_logprobs == [-0.3, -0.7]


def test_rank_single_candidate():
    table = {(10,): [-0.5]}
    out = rank_select([1], [cand([10], index=0)], k=1, score_fn=fixed_scorer(table))
    assert len(out.selected) == 1
    assert out.selected[0].choice_index == 0


def test_rank_metadata_carried_to_steps():
    table = {(10,): [-0.5]}
    out = rank_select(
        [1],
        [cand([10], index=0, text=" A", tokens=[" A"])],
        k=1,
        score_fn=fixed_scorer(table),
    )
    assert out.selected[0].text == " A"
    assert out.selected[0].tokens == [" A"]
