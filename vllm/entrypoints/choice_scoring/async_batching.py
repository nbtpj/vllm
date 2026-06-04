# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async batched orchestration for choice scoring & ranking.

Mirrors :mod:`batching` but ``await``\\ s an async ``ScoreBatchFn`` (e.g. one
backed by ``AsyncLLM.generate``). It reuses the exact same pure
planning/finalizing helpers, so sync and async paths cannot diverge in their
selection semantics -- only in how they invoke the scorer.

Two rank drivers are provided:

* :func:`rank_batch_async` -- lock-step: one fused scoring call per step
  across all still-active prompts (a barrier between steps).
* :func:`rank_batch_pipelined_async` -- per-prompt pipelines: every prompt
  runs its own rank loop concurrently, so prompt A can be on step 5 while
  prompt B is still on step 1. With a continuous-batching engine behind the
  scorer this removes the global step barrier entirely. Selection semantics
  are identical (both drive :class:`~.batching._RankState`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from vllm.entrypoints.choice_scoring.batching import (
    _build_score_pairs,
    _finalize_score,
    _RankState,
    broadcast_k,
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


async def rank_batch_pipelined_async(
    prompts_token_ids: Sequence[list[int]],
    candidates_per_prompt: Sequence[Sequence[Candidate]],
    k_per_prompt: int | Sequence[int],
    score_batch_fn: AsyncScoreBatchFn,
    select_by: SelectBy = "mean",
) -> list[RankOutput]:
    """Rank with one independent pipeline per prompt (no global step barrier).

    Each prompt advances through its ``k`` steps as soon as its own previous
    step finishes; the engine behind ``score_batch_fn`` interleaves the
    in-flight scoring requests of all prompts via continuous batching.
    """
    n = len(prompts_token_ids)
    if len(candidates_per_prompt) != n:
        raise ValueError(
            f"{n} prompts but {len(candidates_per_prompt)} candidate lists"
        )
    ks = broadcast_k(k_per_prompt, n)

    async def _run_one(i: int) -> RankOutput:
        # A single-prompt _RankState: identical semantics to the lock-step
        # driver, just advanced independently of the other prompts.
        state = _RankState([prompts_token_ids[i]], [candidates_per_prompt[i]], [ks[i]])
        for step in range(state.max_steps):
            pairs, index_map = state.plan_step(step)
            if not pairs:
                break
            results = await score_batch_fn(pairs)
            check_result_length("rank_batch_pipelined_async", len(results), len(pairs))
            state.apply_step(step, index_map, results, select_by)
        return state.finalize()[0]

    tasks = [asyncio.ensure_future(_run_one(i)) for i in range(n)]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        raise
