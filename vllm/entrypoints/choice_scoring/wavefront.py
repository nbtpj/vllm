# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wavefront (pipelined) rank driver for the offline engine.

:func:`~vllm.entrypoints.choice_scoring.batching.rank_batch` advances all
prompts in lock-step: step ``s+1`` starts only after *every* prompt finished
step ``s``, so a prompt with a small/short candidate pool idles while the
slowest prompt of the round is still scoring. This driver removes that
barrier for the blocking offline engine: each prompt's next-step requests are
submitted the moment its own previous step completes, keeping the engine's
continuous-batching queue full until the last prompt drains.

The driver is engine-agnostic and CPU-testable: it talks to the engine
through two injected callables --

* ``submit_fn(context_ids, candidate_ids) -> request_id`` adds one scoring
  request to the engine and returns its id.
* ``poll_fn() -> list[(request_id, ScoredContinuation)]`` advances the engine
  one step and returns the scoring results that finished, in any order.

Selection semantics are identical to the lock-step drivers: every prompt is
advanced by its own single-prompt :class:`~.batching._RankState`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from vllm.entrypoints.choice_scoring.batching import _RankState, broadcast_k
from vllm.entrypoints.choice_scoring.core import ScoredContinuation
from vllm.entrypoints.choice_scoring.params import (
    Candidate,
    RankOutput,
    SelectBy,
)

SubmitFn = Callable[[list[int], list[int]], str]
PollFn = Callable[[], list[tuple[str, ScoredContinuation]]]


def expected_rank_requests(pool_size: int, k: int) -> int:
    """Total scoring requests a rank over ``pool_size`` candidates needs.

    Step ``s`` scores the ``pool_size - s`` remaining candidates, for
    ``min(k, pool_size)`` steps.
    """
    steps = min(k, pool_size)
    return sum(pool_size - s for s in range(steps))


def rank_batch_wavefront(
    prompts_token_ids: Sequence[list[int]],
    candidates_per_prompt: Sequence[Sequence[Candidate]],
    k_per_prompt: int | Sequence[int],
    submit_fn: SubmitFn,
    poll_fn: PollFn,
    select_by: SelectBy = "mean",
    abort_fn: Callable[[list[str]], None] | None = None,
    progress_fn: Callable[[int], None] | None = None,
) -> list[RankOutput]:
    """Rank many prompts with per-prompt pipelining over a blocking engine.

    ``abort_fn`` (if given) is called with the still-pending request ids when
    an error escapes, so the engine does not keep scoring orphaned requests.
    ``progress_fn`` (if given) is called with the number of newly finished
    scoring requests after every poll.
    """
    n = len(prompts_token_ids)
    if len(candidates_per_prompt) != n:
        raise ValueError(
            f"{n} prompts but {len(candidates_per_prompt)} candidate lists"
        )
    ks = broadcast_k(k_per_prompt, n)

    states = [
        _RankState([prompts_token_ids[i]], [candidates_per_prompt[i]], [ks[i]])
        for i in range(n)
    ]
    # Per-prompt in-flight step bookkeeping.
    steps = [0] * n
    index_maps: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    buffers: list[list[ScoredContinuation | None]] = [[] for _ in range(n)]
    outstanding = [0] * n
    # request_id -> (prompt index, slot in that prompt's current buffer)
    pending: dict[str, tuple[int, int]] = {}

    def submit_step(i: int, step: int) -> None:
        pairs, index_map = states[i].plan_step(step)
        if not pairs:
            return
        steps[i] = step
        index_maps[i] = index_map
        buffers[i] = [None] * len(pairs)
        outstanding[i] = len(pairs)
        for slot, (ctx, cand) in enumerate(pairs):
            request_id = submit_fn(ctx, cand)
            pending[request_id] = (i, slot)

    for i in range(n):
        submit_step(i, 0)

    try:
        while pending:
            finished = poll_fn()
            for request_id, scored in finished:
                entry = pending.pop(request_id, None)
                if entry is None:
                    continue
                i, slot = entry
                buffers[i][slot] = scored
                outstanding[i] -= 1
                if outstanding[i] == 0:
                    # This prompt's step is complete: select and immediately
                    # submit its next step (no cross-prompt barrier).
                    states[i].apply_step(
                        steps[i],
                        index_maps[i],
                        buffers[i],  # type: ignore[arg-type]
                        select_by,
                    )
                    submit_step(i, steps[i] + 1)
            if progress_fn is not None and finished:
                progress_fn(len(finished))
    except BaseException:
        if abort_fn is not None and pending:
            abort_fn(list(pending.keys()))
        raise

    return [state.finalize()[0] for state in states]
