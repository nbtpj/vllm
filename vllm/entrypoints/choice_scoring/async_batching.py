# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async batched orchestration for choice scoring & ranking.

Mirrors :mod:`batching` but ``await``\\ s an async ``ScoreBatchFn`` (e.g. one
backed by ``AsyncLLM.generate``). It reuses the exact same pure
planning/finalizing helpers, so sync and async paths cannot diverge in their
selection semantics -- only in how they invoke the scorer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from vllm.entrypoints.choice_scoring.batching import (
    _RankState,
    _build_score_pairs,
    _finalize_score,
    check_result_length,
)
from vllm.entrypoints.choice_scoring.core import ScoredContinuation
from vllm.entrypoints.choice_scoring.params import (
    Candidate,
    RankOutput,
    ScoreChoicesOutput,
    SelectBy,
)

# Async scorer over a batch of (context_ids, candidate_ids) pairs.
AsyncScoreBatchFn = Callable[
    [list[tuple[list[int], list[int]]]],
    Awaitable[list[ScoredContinuation]],
]


async def score_choices_batch_async(
    prompts_token_ids: Sequence[list[int]],
    choices_per_prompt: Sequence[Sequence[Candidate]],
    score_batch_fn: AsyncScoreBatchFn,
    select_by: SelectBy = "mean",
) -> list[ScoreChoicesOutput]:
    pairs, spans = _build_score_pairs(prompts_token_ids, choices_per_prompt)
    results = await score_batch_fn(pairs)
    check_result_length("score_choices_batch_async", len(results), len(pairs))
    return _finalize_score(spans, results, select_by)


async def rank_batch_async(
    prompts_token_ids: Sequence[list[int]],
    candidates_per_prompt: Sequence[Sequence[Candidate]],
    k_per_prompt: int | Sequence[int],
    score_batch_fn: AsyncScoreBatchFn,
    select_by: SelectBy = "mean",
) -> list[RankOutput]:
    state = _RankState(prompts_token_ids, candidates_per_prompt, k_per_prompt)
    for step in range(state.max_steps):
        pairs, index_map = state.plan_step(step)
        if not pairs:
            break
        results = await score_batch_fn(pairs)
        check_result_length("rank_batch_async", len(results), len(pairs))
        state.apply_step(step, index_map, results, select_by)
    return state.finalize()
