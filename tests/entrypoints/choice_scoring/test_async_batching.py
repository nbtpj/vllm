# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the async batched orchestration (no model / GPU).

Uses ``asyncio.run`` (no pytest-asyncio dependency) with an async fake scorer,
and cross-checks that the async driver produces identical results to the sync
driver for the same inputs -- proving the shared helpers keep them in lockstep.
"""

from __future__ import annotations

import asyncio

import pytest

from vllm.entrypoints.choice_scoring.async_batching import (
    rank_batch_async,
    score_choices_batch_async,
)
from vllm.entrypoints.choice_scoring.batching import (
    rank_batch,
    score_choices_batch,
)
from vllm.entrypoints.choice_scoring.params import Candidate


def cand(token_ids, index=0):
    return Candidate(token_ids=token_ids, index=index)


def sync_table_fn(table):
    def _fn(pairs):
        return [(list(table[tuple(c)]), None) for _ctx, c in pairs]

    return _fn


def async_table_fn(table, record=None):
    async def _fn(pairs):
        if record is not None:
            record.append(len(pairs))
        # Simulate a backend hop.
        await asyncio.sleep(0)
        return [(list(table[tuple(c)]), None) for _ctx, c in pairs]

    return _fn


TABLE = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0], (20,): [-0.5], (21,): [-0.9]}
PROMPTS = [[1], [2]]
CHOICES = [
    [cand([10], 0), cand([11], 1), cand([12], 2)],
    [cand([20], 0), cand([21], 1)],
]


def test_async_score_matches_sync():
    async_out = asyncio.run(
        score_choices_batch_async(PROMPTS, CHOICES, async_table_fn(TABLE))
    )
    sync_out = score_choices_batch(PROMPTS, CHOICES, sync_table_fn(TABLE))
    assert [o.best_choice_index for o in async_out] == [
        o.best_choice_index for o in sync_out
    ]
    for ao, so in zip(async_out, sync_out):
        assert [c.token_logprobs for c in ao.choices] == [
            c.token_logprobs for c in so.choices
        ]


def test_async_rank_matches_sync():
    async_out = asyncio.run(
        rank_batch_async(PROMPTS, CHOICES, [3, 2], async_table_fn(TABLE))
    )
    sync_out = rank_batch(PROMPTS, CHOICES, [3, 2], sync_table_fn(TABLE))
    for ao, so in zip(async_out, sync_out):
        assert [s.choice_index for s in ao.selected] == [
            s.choice_index for s in so.selected
        ]
        assert ao.truncated == so.truncated


def test_async_rank_batches_each_step_across_prompts():
    record = []
    asyncio.run(
        rank_batch_async(PROMPTS, CHOICES, [3, 1], async_table_fn(TABLE, record=record))
    )
    # Step 0: 3 + 2 = 5 pairs; step 1: prompt0 has 2 left; step 2: 1 left.
    assert record == [5, 2, 1]


def test_async_score_result_length_mismatch_raises():
    async def bad(pairs):
        return []

    with pytest.raises(ValueError, match="returned 0 results"):
        asyncio.run(score_choices_batch_async([[1]], [[cand([10], 0)]], bad))
