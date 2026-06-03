# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure orchestration logic for choice scoring and ranking.

Everything here is model-agnostic: callers inject a ``score_fn`` that, given a
context (list of token ids) and a batch of candidate token-id lists, returns
the teacher-forced per-token logprobs (and optional ranks) of each candidate
*conditioned on that context*. This separation makes every aggregation and
selection rule deterministically testable without a GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from vllm.entrypoints.choice_scoring.params import (
    Candidate,
    ChoiceScore,
    RankOutput,
    RankStep,
    ScoreChoicesOutput,
    SelectBy,
)

# Per-candidate teacher-forced result returned by a ``score_fn``:
#   (token_logprobs, ranks)
# ``ranks`` may be ``None`` if the engine was not asked for vocab ranks. When
# present it must align 1:1 with ``token_logprobs`` and use 1-based ranks
# (rank == 1 means the token was the argmax at that position).
ScoredContinuation = tuple[list[float], list[int] | None]

# Signature of the injected scorer. Implementations must preserve input order:
# the i-th returned result corresponds to ``candidate_token_ids[i]``.
ScoreFn = Callable[[list[int], list[list[int]]], list[ScoredContinuation]]


def aggregate_choice(
    candidate: Candidate,
    token_logprobs: list[float],
    ranks: list[int] | None = None,
) -> ChoiceScore:
    """Aggregate a candidate's per-token logprobs into a :class:`ChoiceScore`.

    Raises:
        ValueError: if the candidate has no tokens, or if ``token_logprobs`` /
            ``ranks`` lengths disagree with the candidate's token count.
    """
    n = len(candidate.token_ids)
    if n == 0:
        raise ValueError(
            f"choice {candidate.index} has zero tokens; cannot score an "
            "empty continuation"
        )
    if len(token_logprobs) != n:
        raise ValueError(
            f"choice {candidate.index}: got {len(token_logprobs)} logprobs "
            f"for {n} tokens"
        )
    if ranks is not None and len(ranks) != n:
        raise ValueError(
            f"choice {candidate.index}: got {len(ranks)} ranks for {n} tokens"
        )

    total = float(sum(token_logprobs))
    is_greedy = None if ranks is None else all(r == 1 for r in ranks)
    return ChoiceScore(
        index=candidate.index,
        token_ids=list(candidate.token_ids),
        token_logprobs=list(token_logprobs),
        sum_logprob=total,
        mean_logprob=total / n,
        is_greedy=is_greedy,
        tokens=candidate.tokens,
        text=candidate.text,
    )


def _metric(choice: ChoiceScore, select_by: SelectBy) -> float:
    if select_by == "mean":
        return choice.mean_logprob
    if select_by == "sum":
        return choice.sum_logprob
    raise ValueError(f"unknown select_by={select_by!r}; expected 'mean' or 'sum'")


def select_best_index(
    choices: Sequence[ChoiceScore],
    select_by: SelectBy = "mean",
) -> int:
    """Return the position of the best choice, with lowest-index tie-breaking.

    Returns ``-1`` for an empty input.
    """
    best_pos = -1
    best_val = float("-inf")
    for pos, choice in enumerate(choices):
        val = _metric(choice, select_by)
        # Strict ``>`` keeps the earliest candidate on ties.
        if val > best_val:
            best_val = val
            best_pos = pos
    return best_pos


def _normalize_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Validate the pool and stamp stable indices (0..n-1)."""
    if len(candidates) == 0:
        raise ValueError("at least one choice/candidate is required")
    normalized: list[Candidate] = []
    for i, cand in enumerate(candidates):
        if len(cand.token_ids) == 0:
            raise ValueError(
                f"choice at position {i} has zero tokens; every choice must "
                "tokenize to at least one token"
            )
        normalized.append(
            Candidate(
                token_ids=list(cand.token_ids),
                text=cand.text,
                tokens=cand.tokens,
                index=i,
            )
        )
    return normalized


def score_choices(
    prompt_token_ids: list[int],
    candidates: Sequence[Candidate],
    score_fn: ScoreFn,
    select_by: SelectBy = "mean",
) -> ScoreChoicesOutput:
    """Score every choice against a single fixed prompt.

    Each choice is scored independently (teacher forced) against the same
    ``prompt_token_ids`` context.
    """
    cands = _normalize_candidates(candidates)
    scored = score_fn(list(prompt_token_ids), [c.token_ids for c in cands])
    if len(scored) != len(cands):
        raise ValueError(
            f"score_fn returned {len(scored)} results for {len(cands)} choices"
        )

    choices = [
        aggregate_choice(cand, logprobs, ranks)
        for cand, (logprobs, ranks) in zip(cands, scored)
    ]
    best = select_best_index(choices, select_by)
    return ScoreChoicesOutput(
        prompt_token_ids=list(prompt_token_ids),
        choices=choices,
        best_choice_index=best,
    )


def rank_select(
    prompt_token_ids: list[int],
    candidates: Sequence[Candidate],
    k: int,
    score_fn: ScoreFn,
    select_by: SelectBy = "mean",
) -> RankOutput:
    """Autoregressively select an ordered list of ``k`` bundles.

    At each step every *remaining* candidate is re-scored against the current
    context (``prompt`` + already-selected bundles), the best by ``select_by``
    is appended to the context and removed from the pool, and the loop repeats.

    If ``k`` exceeds the pool size, selection stops when the pool is exhausted
    and ``RankOutput.truncated`` is set to ``True``.

    Raises:
        ValueError: if ``k`` is negative or the candidate pool is empty.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    cands = _normalize_candidates(candidates)

    truncated = k > len(cands)
    steps = min(k, len(cands))

    context = list(prompt_token_ids)
    remaining = list(cands)
    selected: list[RankStep] = []

    for order in range(steps):
        scored = score_fn(context, [c.token_ids for c in remaining])
        if len(scored) != len(remaining):
            raise ValueError(
                f"score_fn returned {len(scored)} results for "
                f"{len(remaining)} remaining candidates"
            )
        choice_scores = [
            aggregate_choice(cand, logprobs, ranks)
            for cand, (logprobs, ranks) in zip(remaining, scored)
        ]
        best_pos = select_best_index(choice_scores, select_by)
        best_choice = choice_scores[best_pos]
        best_cand = remaining[best_pos]

        selected.append(
            RankStep(
                order=order,
                choice_index=best_choice.index,
                token_ids=best_choice.token_ids,
                token_logprobs=best_choice.token_logprobs,
                sum_logprob=best_choice.sum_logprob,
                mean_logprob=best_choice.mean_logprob,
                tokens=best_cand.tokens,
                text=best_cand.text,
            )
        )
        # Autoregressive conditioning + pool shrink.
        context = context + best_cand.token_ids
        remaining.pop(best_pos)

    return RankOutput(
        prompt_token_ids=list(prompt_token_ids),
        selected=selected,
        truncated=truncated,
    )
