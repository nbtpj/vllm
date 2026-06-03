# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for batched score/rank orchestration (no model / GPU).

A deterministic in-memory ``ScoreBatchFn`` lets us verify cross-prompt
batching, per-step rank batching, regrouping, per-prompt independence,
autoregressive context growth and truncation -- the parts most likely to have
indexing bugs.
"""

from __future__ import annotations

import pytest

from vllm.entrypoints.choice_scoring.batching import (
    rank_batch,
    score_choices_batch,
)
from vllm.entrypoints.choice_scoring.params import Candidate


def cand(token_ids, index=0, text=None, tokens=None):
    return Candidate(token_ids=token_ids, index=index, text=text, tokens=tokens)


def table_batch_fn(table, ranks_table=None, record=None):
    """ScoreBatchFn: look each (cand tuple) up in ``table`` (context-free)."""

    def _fn(pairs):
        if record is not None:
            record.append([(list(ctx), list(c)) for ctx, c in pairs])
        out = []
        for _ctx, c in pairs:
            key = tuple(c)
            ranks = None if ranks_table is None else ranks_table[key]
            out.append((list(table[key]), ranks))
        return out

    return _fn


def context_batch_fn(score_of, record=None):
    """ScoreBatchFn whose scores depend on the context's last token."""

    def _fn(pairs):
        if record is not None:
            record.append([(list(ctx), list(c)) for ctx, c in pairs])
        last_of = lambda ctx: ctx[-1] if ctx else None
        return [(list(score_of(last_of(ctx), c)), None) for ctx, c in pairs]

    return _fn


# --------------------------------------------------------------------------- #
# score_choices_batch
# --------------------------------------------------------------------------- #
def test_score_batch_multiple_prompts_independent():
    table = {
        (10,): [-1.0],
        (11,): [-2.0],
        (20,): [-0.5],
        (21, 22): [-0.1, -0.1],
    }
    prompts = [[1, 2], [3]]
    choices = [
        [cand([10], index=0), cand([11], index=1)],
        [cand([20], index=0), cand([21, 22], index=1)],
    ]
    outs = score_choices_batch(prompts, choices, table_batch_fn(table))
    assert len(outs) == 2
    # Prompt 0: mean -1.0 vs -2.0 -> choice 0
    assert outs[0].best_choice_index == 0
    assert outs[0].prompt_token_ids == [1, 2]
    # Prompt 1: mean -0.5 vs -0.1 -> choice 1
    assert outs[1].best_choice_index == 1
    assert len(outs[1].choices) == 2


def test_score_batch_flattens_into_single_call():
    record = []
    table = {(10,): [-1.0], (11,): [-2.0], (20,): [-0.5]}
    prompts = [[1], [2]]
    choices = [[cand([10], index=0), cand([11], index=1)], [cand([20], index=0)]]
    score_choices_batch(prompts, choices, table_batch_fn(table, record=record))
    # Exactly one batched call containing all 3 (prompt, choice) pairs.
    assert len(record) == 1
    assert len(record[0]) == 3
    # Contexts are the right prompts.
    assert record[0][0][0] == [1]
    assert record[0][2][0] == [2]


def test_score_batch_prompt_choice_count_mismatch_raises():
    with pytest.raises(ValueError, match="prompts but"):
        score_choices_batch([[1], [2]], [[cand([10])]], table_batch_fn({}))


def test_score_batch_scorer_wrong_length_raises():
    def bad(pairs):
        return []

    with pytest.raises(ValueError, match="returned 0 results"):
        score_choices_batch([[1]], [[cand([10], index=0)]], bad)


def test_score_batch_empty_choices_for_a_prompt_raises():
    with pytest.raises(ValueError, match="at least one"):
        score_choices_batch([[1]], [[]], table_batch_fn({}))


# --------------------------------------------------------------------------- #
# rank_batch
# --------------------------------------------------------------------------- #
def _cands(ids_and_idx):
    return [cand(ids, index=i) for i, ids in enumerate(ids_and_idx)]


def test_rank_batch_per_prompt_independent_orders():
    table = {
        # prompt 0 pool
        (10,): [-3.0],
        (11,): [-1.0],
        (12,): [-2.0],
        # prompt 1 pool
        (20,): [-0.5],
        (21,): [-0.9],
    }
    prompts = [[1], [2]]
    cands = [
        [cand([10], 0), cand([11], 1), cand([12], 2)],
        [cand([20], 0), cand([21], 1)],
    ]
    outs = rank_batch(
        prompts, cands, k_per_prompt=[3, 2], score_batch_fn=table_batch_fn(table)
    )
    assert [s.choice_index for s in outs[0].selected] == [1, 2, 0]
    assert [s.choice_index for s in outs[1].selected] == [0, 1]
    assert outs[0].truncated is False and outs[1].truncated is False


def test_rank_batch_k_broadcast_int():
    table = {(10,): [-1.0], (11,): [-2.0]}
    outs = rank_batch(
        [[1], [2]],
        [[cand([10], 0), cand([11], 1)], [cand([10], 0), cand([11], 1)]],
        k_per_prompt=1,
        score_batch_fn=table_batch_fn(table),
    )
    assert all(len(o.selected) == 1 for o in outs)
    assert all(o.selected[0].choice_index == 0 for o in outs)


def test_rank_batch_differing_k_and_truncation():
    table = {(10,): [-1.0], (11,): [-2.0]}
    outs = rank_batch(
        [[1], [2]],
        [[cand([10], 0), cand([11], 1)], [cand([10], 0), cand([11], 1)]],
        k_per_prompt=[1, 5],  # second exceeds pool of 2
        score_batch_fn=table_batch_fn(table),
    )
    assert len(outs[0].selected) == 1
    assert outs[0].truncated is False
    assert len(outs[1].selected) == 2  # exhausted at 2
    assert outs[1].truncated is True


def test_rank_batch_k_zero_prompt_contributes_nothing():
    record = []
    table = {(10,): [-1.0], (11,): [-2.0], (20,): [-0.5]}
    outs = rank_batch(
        [[1], [2]],
        [[cand([10], 0), cand([11], 1)], [cand([20], 0)]],
        k_per_prompt=[0, 1],
        score_batch_fn=table_batch_fn(table, record=record),
    )
    assert outs[0].selected == []
    assert len(outs[1].selected) == 1
    # Step 0 batch must contain only prompt 1's single candidate.
    assert len(record[0]) == 1
    assert record[0][0][0] == [2]


def test_rank_batch_step_batches_only_active_remaining():
    record = []
    table = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0], (20,): [-0.5], (21,): [-0.9]}
    rank_batch(
        [[1], [2]],
        [
            [cand([10], 0), cand([11], 1), cand([12], 2)],  # k=3
            [cand([20], 0), cand([21], 1)],  # k=1
        ],
        k_per_prompt=[3, 1],
        score_batch_fn=table_batch_fn(table, record=record),
    )
    # Step 0: prompt0 has 3, prompt1 has 2 -> 5 pairs.
    assert len(record[0]) == 5
    # Step 1: prompt0 has 2 remaining, prompt1 done -> 2 pairs.
    assert len(record[1]) == 2
    # Step 2: prompt0 has 1 remaining -> 1 pair.
    assert len(record[2]) == 1


def test_rank_batch_autoregressive_context_growth_per_prompt():
    record = []
    table = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0]}
    rank_batch(
        [[7, 8]],
        [[cand([10], 0), cand([11], 1), cand([12], 2)]],
        k_per_prompt=3,
        score_batch_fn=table_batch_fn(table, record=record),
    )
    # Best order is 11, 12, 10. Context grows accordingly.
    assert record[0][0][0] == [7, 8]
    assert record[1][0][0] == [7, 8, 11]
    assert record[2][0][0] == [7, 8, 11, 12]


def test_rank_batch_autoregressive_changes_outcome():
    # Mirror the single-prompt autoregressive test at batch level.
    def score_of(last, c):
        first = c[0]
        if last == 0:
            return [{10: -1.0, 11: -2.0, 12: -3.0}[first]]
        if last == 10:
            return [{11: -5.0, 12: -0.5}[first]]
        if last == 12:
            return [{11: -0.1}[first]]
        raise AssertionError(last)

    outs = rank_batch(
        [[0]],
        [[cand([10], 0), cand([11], 1), cand([12], 2)]],
        k_per_prompt=3,
        score_batch_fn=context_batch_fn(score_of),
    )
    assert [s.choice_index for s in outs[0].selected] == [0, 2, 1]


def test_rank_batch_no_duplicates_within_prompt():
    table = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0]}
    outs = rank_batch(
        [[1]],
        [[cand([10], 0), cand([11], 1), cand([12], 2)]],
        k_per_prompt=3,
        score_batch_fn=table_batch_fn(table),
    )
    picked = [s.choice_index for s in outs[0].selected]
    assert sorted(picked) == [0, 1, 2]


def test_rank_batch_negative_k_raises():
    with pytest.raises(ValueError, match="k must be >= 0"):
        rank_batch(
            [[1]], [[cand([10], 0)]], k_per_prompt=-1, score_batch_fn=table_batch_fn({})
        )


def test_rank_batch_k_list_wrong_length_raises():
    with pytest.raises(ValueError, match="expected 1 k values"):
        rank_batch(
            [[1]],
            [[cand([10], 0)]],
            k_per_prompt=[1, 2],
            score_batch_fn=table_batch_fn({}),
        )


def test_rank_batch_prompt_candidate_count_mismatch_raises():
    with pytest.raises(ValueError, match="prompts but"):
        rank_batch(
            [[1], [2]],
            [[cand([10], 0)]],
            k_per_prompt=1,
            score_batch_fn=table_batch_fn({}),
        )


def test_rank_batch_tie_breaks_lowest_index_per_prompt():
    table = {(10,): [-1.0], (11,): [-1.0], (12,): [-1.0]}
    outs = rank_batch(
        [[1]],
        [[cand([10], 0), cand([11], 1), cand([12], 2)]],
        k_per_prompt=3,
        score_batch_fn=table_batch_fn(table),
    )
    assert [s.choice_index for s in outs[0].selected] == [0, 1, 2]
