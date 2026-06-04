# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the offline wavefront rank driver (no model / GPU).

A fake engine hands results back through ``poll_fn`` in controlled (and
deliberately adversarial) orders, proving:

* parity with the lock-step ``rank_batch`` driver,
* per-prompt pipelining (a prompt advances to its next step while another
  prompt's requests are still pending),
* abort of in-flight requests when an error escapes,
* progress reporting.
"""

from __future__ import annotations

import pytest

from vllm.entrypoints.choice_scoring.batching import rank_batch
from vllm.entrypoints.choice_scoring.params import Candidate
from vllm.entrypoints.choice_scoring.wavefront import (
    expected_rank_requests,
    rank_batch_wavefront,
)


def cand(token_ids, index=0):
    return Candidate(token_ids=token_ids, index=index)


TABLE = {(10,): [-3.0], (11,): [-1.0], (12,): [-2.0], (20,): [-0.5], (21,): [-0.9]}
PROMPTS = [[1], [2]]
CHOICES = [
    [cand([10], 0), cand([11], 1), cand([12], 2)],
    [cand([20], 0), cand([21], 1)],
]


class FakeEngine:
    """Queues submitted requests; finishes them via an injectable policy."""

    def __init__(self, table, finish_order=None, per_poll=None):
        self.table = table
        self.queue: list[tuple[str, tuple]] = []  # (request_id, cand)
        self.finish_order = finish_order  # callable(queue) -> index to pop
        self.per_poll = per_poll  # max results returned per poll
        self.submitted: list[str] = []
        self.aborted: list[str] = []

    def submit(self, ctx, cand):
        request_id = f"req-{len(self.submitted)}"
        self.submitted.append(request_id)
        self.queue.append((request_id, tuple(cand)))
        return request_id

    def poll(self):
        results = []
        budget = self.per_poll or len(self.queue)
        while self.queue and len(results) < budget:
            idx = self.finish_order(self.queue) if self.finish_order else 0
            request_id, cand = self.queue.pop(idx)
            results.append((request_id, (list(self.table[cand]), None)))
        return results

    def abort(self, request_ids):
        self.aborted.extend(request_ids)
        self.queue = [(r, c) for r, c in self.queue if r not in request_ids]


def selected_indices(outputs):
    return [[s.choice_index for s in o.selected] for o in outputs]


def test_wavefront_matches_lockstep():
    engine = FakeEngine(TABLE)
    out = rank_batch_wavefront(PROMPTS, CHOICES, [3, 2], engine.submit, engine.poll)

    def sync_fn(pairs):
        return [(list(TABLE[tuple(c)]), None) for _ctx, c in pairs]

    lockstep = rank_batch(PROMPTS, CHOICES, [3, 2], sync_fn)
    assert selected_indices(out) == selected_indices(lockstep)
    assert [o.truncated for o in out] == [o.truncated for o in lockstep]


def test_wavefront_out_of_order_completion():
    # Finish requests LIFO and one per poll: results arrive in the most
    # adversarial order, slots must still land correctly.
    engine = FakeEngine(TABLE, finish_order=lambda q: len(q) - 1, per_poll=1)
    out = rank_batch_wavefront(PROMPTS, CHOICES, [3, 2], engine.submit, engine.poll)
    assert selected_indices(out) == [[1, 2, 0], [0, 1]]


def test_wavefront_prompts_advance_independently():
    """Finish only prompt B's requests first: B must reach step 1 while all
    of prompt A's step-0 requests are still pending."""
    engine = FakeEngine(TABLE)
    b_steps_started = []

    orig_submit = engine.submit

    def tracking_submit(ctx, cand):
        if ctx[0] == 2:  # prompt B (context starts with its prompt token 2)
            b_steps_started.append(len(ctx))
        return orig_submit(ctx, cand)

    def finish_b_first(queue):
        for idx, (_rid, cand) in enumerate(queue):
            if cand[0] >= 20:  # B's candidates are 20/21
                return idx
        return 0

    engine.finish_order = finish_b_first
    engine.per_poll = 1
    out = rank_batch_wavefront(PROMPTS, CHOICES, [3, 2], tracking_submit, engine.poll)
    # B's step-1 submission (context length 2 = prompt + selected bundle)
    # happened; A's pool was never blocked on it and the result is complete.
    assert 2 in b_steps_started
    assert selected_indices(out) == [[1, 2, 0], [0, 1]]


def test_wavefront_aborts_pending_on_error():
    engine = FakeEngine(TABLE)

    polls = [0]

    def exploding_poll():
        polls[0] += 1
        if polls[0] == 1:
            # Finish exactly one request, then explode on the next poll.
            return engine.poll()[:1]
        raise RuntimeError("engine crashed")

    engine.per_poll = 1
    with pytest.raises(RuntimeError, match="engine crashed"):
        rank_batch_wavefront(
            PROMPTS,
            CHOICES,
            [3, 2],
            engine.submit,
            exploding_poll,
            abort_fn=engine.abort,
        )
    # All step-0 requests were submitted (3 + 2); one finished, the other
    # four must have been aborted.
    assert len(engine.submitted) == 5
    assert len(engine.aborted) == 4


def test_wavefront_progress_and_expected_counts():
    engine = FakeEngine(TABLE)
    ticks = []
    out = rank_batch_wavefront(
        PROMPTS,
        CHOICES,
        [3, 2],
        engine.submit,
        engine.poll,
        progress_fn=ticks.append,
    )
    total = sum(ticks)
    assert total == expected_rank_requests(3, 3) + expected_rank_requests(2, 2)
    assert total == len(engine.submitted)
    assert [o.truncated for o in out] == [False, False]


def test_wavefront_k_zero_submits_nothing():
    engine = FakeEngine(TABLE)
    out = rank_batch_wavefront(PROMPTS, CHOICES, [0, 0], engine.submit, engine.poll)
    assert [o.selected for o in out] == [[], []]
    assert [o.truncated for o in out] == [False, False]
    assert engine.submitted == []


def test_wavefront_empty_pool_rejected_like_lockstep():
    engine = FakeEngine(TABLE)
    with pytest.raises(ValueError, match="at least one choice"):
        rank_batch_wavefront([[1]], [[]], [1], engine.submit, engine.poll)


def test_expected_rank_requests():
    assert expected_rank_requests(3, 3) == 3 + 2 + 1
    assert expected_rank_requests(3, 2) == 3 + 2
    assert expected_rank_requests(2, 5) == 2 + 1
    assert expected_rank_requests(0, 4) == 0
