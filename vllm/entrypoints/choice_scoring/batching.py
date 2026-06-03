# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched orchestration for choice scoring & ranking.

This layer turns the single-prompt :mod:`core` logic into efficient batched
operations:

* :func:`score_choices_batch` flattens every ``(prompt, choice)`` pair across
  the whole batch into a single scoring call so the backend can fuse them (and
  reuse the shared-prompt KV cache).
* :func:`rank_batch` drives the autoregressive rank loop for many prompts at
  once: at each step it gathers the *remaining* candidates of every still-active
  prompt into one scoring call, then advances each prompt independently. Prompts
  with smaller ``k`` (or smaller pools) simply drop out of later steps.

Both are parameterised by a ``ScoreBatchFn`` -- a callable that scores a batch
of ``(context_ids, candidate_ids)`` pairs. The pure step planning/applying
helpers (``_build_score_pairs``, ``_finalize_score``, ``_init_rank_state``,
``_plan_rank_step``, ``_apply_rank_step``, ``_finalize_rank``) are shared with
the async driver in :mod:`async_batching` so the two cannot diverge.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from vllm.entrypoints.choice_scoring.core import (
    ScoredContinuation,
    aggregate_choice,
    select_best_index,
)
from vllm.entrypoints.choice_scoring.core import (
    _normalize_candidates as _normalize,
)
from vllm.entrypoints.choice_scoring.params import (
    Candidate,
    RankOutput,
    RankStep,
    ScoreChoicesOutput,
    SelectBy,
)

# A scorer over a heterogeneous batch of (context_ids, candidate_ids) pairs.
# Must return one ScoredContinuation per input pair, in the same order.
ScoreBatchFn = Callable[[list[tuple[list[int], list[int]]]], list[ScoredContinuation]]

# (start, end, normalized_candidates, prompt_token_ids) per prompt.
_ScoreSpan = tuple[int, int, list[Candidate], list[int]]


def check_result_length(name: str, got: int, want: int) -> None:
    if got != want:
        raise ValueError(f"{name}: scorer returned {got} results for {want} pairs")


# --------------------------------------------------------------------------- #
# score helpers (shared sync/async)
# --------------------------------------------------------------------------- #
def _build_score_pairs(
    prompts_token_ids: Sequence[list[int]],
    choices_per_prompt: Sequence[Sequence[Candidate]],
) -> tuple[list[tuple[list[int], list[int]]], list[_ScoreSpan]]:
    if len(prompts_token_ids) != len(choices_per_prompt):
        raise ValueError(
            f"{len(prompts_token_ids)} prompts but "
            f"{len(choices_per_prompt)} choice lists"
        )
    pairs: list[tuple[list[int], list[int]]] = []
    spans: list[_ScoreSpan] = []
    for prompt, choices in zip(prompts_token_ids, choices_per_prompt):
        cands = _normalize(choices)
        start = len(pairs)
        for cand in cands:
            pairs.append((list(prompt), cand.token_ids))
        spans.append((start, len(pairs), cands, list(prompt)))
    return pairs, spans


def _finalize_score(
    spans: list[_ScoreSpan],
    results: list[ScoredContinuation],
    select_by: SelectBy,
) -> list[ScoreChoicesOutput]:
    outputs: list[ScoreChoicesOutput] = []
    for start, end, cands, prompt in spans:
        segment = results[start:end]
        scored = [
            aggregate_choice(cand, logprobs, ranks)
            for cand, (logprobs, ranks) in zip(cands, segment)
        ]
        outputs.append(
            ScoreChoicesOutput(
                prompt_token_ids=prompt,
                choices=scored,
                best_choice_index=select_best_index(scored, select_by),
            )
        )
    return outputs


def score_choices_batch(
    prompts_token_ids: Sequence[list[int]],
    choices_per_prompt: Sequence[Sequence[Candidate]],
    score_batch_fn: ScoreBatchFn,
    select_by: SelectBy = "mean",
) -> list[ScoreChoicesOutput]:
    """Score every choice of every prompt in a single fused scoring call."""
    pairs, spans = _build_score_pairs(prompts_token_ids, choices_per_prompt)
    results = score_batch_fn(pairs)
    check_result_length("score_choices_batch", len(results), len(pairs))
    return _finalize_score(spans, results, select_by)


# --------------------------------------------------------------------------- #
# rank helpers (shared sync/async)
# --------------------------------------------------------------------------- #
def broadcast_k(k: int | Sequence[int], n: int) -> list[int]:
    if isinstance(k, int):
        return [k] * n
    ks = list(k)
    if len(ks) != n:
        raise ValueError(f"expected {n} k values, got {len(ks)}")
    return ks


class _RankState:
    """Mutable per-batch ranking state shared by sync/async drivers."""

    def __init__(
        self,
        prompts_token_ids: Sequence[list[int]],
        candidates_per_prompt: Sequence[Sequence[Candidate]],
        k_per_prompt: int | Sequence[int],
    ) -> None:
        n = len(prompts_token_ids)
        if len(candidates_per_prompt) != n:
            raise ValueError(
                f"{n} prompts but {len(candidates_per_prompt)} candidate lists"
            )
        ks = broadcast_k(k_per_prompt, n)
        for k in ks:
            if k < 0:
                raise ValueError(f"k must be >= 0, got {k}")

        self.n = n
        self.prompts = [list(p) for p in prompts_token_ids]
        self.contexts = [list(p) for p in prompts_token_ids]
        self.remaining = [_normalize(c) for c in candidates_per_prompt]
        self.truncated = [ks[i] > len(self.remaining[i]) for i in range(n)]
        self.targets = [min(ks[i], len(self.remaining[i])) for i in range(n)]
        self.selected: list[list[RankStep]] = [[] for _ in range(n)]

    @property
    def max_steps(self) -> int:
        return max(self.targets) if self.targets else 0

    def plan_step(
        self, step: int
    ) -> tuple[list[tuple[list[int], list[int]]], list[tuple[int, int]]]:
        pairs: list[tuple[list[int], list[int]]] = []
        index_map: list[tuple[int, int]] = []
        for i in range(self.n):
            if step < self.targets[i]:
                for j, cand in enumerate(self.remaining[i]):
                    pairs.append((self.contexts[i], cand.token_ids))
                    index_map.append((i, j))
        return pairs, index_map

    def apply_step(
        self,
        step: int,
        index_map: list[tuple[int, int]],
        results: list[ScoredContinuation],
        select_by: SelectBy,
    ) -> None:
        # Group scored candidates per prompt, preserving remaining order.
        grouped: dict[int, list[tuple[int, object]]] = {}
        for (i, j), (logprobs, ranks) in zip(index_map, results):
            cs = aggregate_choice(self.remaining[i][j], logprobs, ranks)
            grouped.setdefault(i, []).append((j, cs))

        for i, items in grouped.items():
            scores = [cs for _, cs in items]
            best_pos = select_best_index(scores, select_by)
            j_best = items[best_pos][0]
            best_cs = items[best_pos][1]
            best_cand = self.remaining[i][j_best]
            self.selected[i].append(
                RankStep(
                    order=step,
                    choice_index=best_cs.index,
                    token_ids=best_cs.token_ids,
                    token_logprobs=best_cs.token_logprobs,
                    sum_logprob=best_cs.sum_logprob,
                    mean_logprob=best_cs.mean_logprob,
                    tokens=best_cand.tokens,
                    text=best_cand.text,
                )
            )
            self.contexts[i] = self.contexts[i] + best_cand.token_ids
            self.remaining[i].pop(j_best)

    def finalize(self) -> list[RankOutput]:
        return [
            RankOutput(
                prompt_token_ids=list(self.prompts[i]),
                selected=self.selected[i],
                truncated=self.truncated[i],
            )
            for i in range(self.n)
        ]


def rank_batch(
    prompts_token_ids: Sequence[list[int]],
    candidates_per_prompt: Sequence[Sequence[Candidate]],
    k_per_prompt: int | Sequence[int],
    score_batch_fn: ScoreBatchFn,
    select_by: SelectBy = "mean",
) -> list[RankOutput]:
    """Autoregressive rank for many prompts, batching each step across prompts."""
    state = _RankState(prompts_token_ids, candidates_per_prompt, k_per_prompt)
    for step in range(state.max_steps):
        pairs, index_map = state.plan_step(step)
        if not pairs:
            break
        results = score_batch_fn(pairs)
        check_result_length("rank_batch", len(results), len(pairs))
        state.apply_step(step, index_map, results, select_by)
    return state.finalize()
