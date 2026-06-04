# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for prompt_logprobs continuation extraction (no model / GPU).

These pin down the position arithmetic and rank handling of the reference
scorer using synthetic ``Logprob`` dicts, plus the batched ``ScoreBatchFn``
wrapper driven by a fake ``generate_fn``.
"""

from __future__ import annotations

import pytest

from vllm.entrypoints.choice_scoring.reference import (
    extract_continuation_logprobs,
    make_prompt_logprobs_score_batch_fn,
)
from vllm.logprobs import Logprob


def lp(logprob, rank=None):
    return Logprob(logprob=logprob, rank=rank)


def make_prompt_logprobs(entries):
    """entries: list of (None | dict[token_id -> (logprob, rank)])."""
    out = []
    for e in entries:
        if e is None:
            out.append(None)
        else:
            out.append({tid: lp(v[0], v[1]) for tid, v in e.items()})
    return out


# --------------------------------------------------------------------------- #
# extract_continuation_logprobs
# --------------------------------------------------------------------------- #
def test_extract_basic_multitoken():
    # context = 2 tokens, candidate = [50, 60] at positions 2 and 3.
    pl = make_prompt_logprobs(
        [
            None,  # pos 0
            {7: (-0.1, 1)},  # pos 1 (still part of context prediction)
            {50: (-0.5, 1)},  # pos 2 -> candidate token 0
            {60: (-1.5, 3)},  # pos 3 -> candidate token 1
        ]
    )
    logprobs, ranks = extract_continuation_logprobs(
        pl, context_len=2, candidate_token_ids=[50, 60]
    )
    assert logprobs == [-0.5, -1.5]
    assert ranks == [1, 3]


def test_extract_single_token():
    pl = make_prompt_logprobs([None, {99: (-2.0, 1)}])
    logprobs, ranks = extract_continuation_logprobs(
        pl, context_len=1, candidate_token_ids=[99]
    )
    assert logprobs == [-2.0]
    assert ranks == [1]


def test_extract_ranks_none_when_missing():
    pl = make_prompt_logprobs([None, {99: (-2.0, None)}])
    logprobs, ranks = extract_continuation_logprobs(
        pl, context_len=1, candidate_token_ids=[99]
    )
    assert logprobs == [-2.0]
    assert ranks is None


def test_extract_ranks_none_if_any_missing():
    pl = make_prompt_logprobs([None, {1: (-1.0, 1)}, {2: (-1.0, None)}])
    _, ranks = extract_continuation_logprobs(
        pl, context_len=1, candidate_token_ids=[1, 2]
    )
    assert ranks is None


def test_extract_none_prompt_logprobs_raises():
    with pytest.raises(ValueError, match="prompt_logprobs is None"):
        extract_continuation_logprobs(None, context_len=1, candidate_token_ids=[1])


def test_extract_empty_candidate_raises():
    with pytest.raises(ValueError, match="zero tokens"):
        extract_continuation_logprobs(make_prompt_logprobs([None]), 1, [])


def test_extract_position_zero_candidate_raises():
    # context_len 0 means candidate starts at absolute position 0.
    pl = make_prompt_logprobs([None, {1: (-1.0, 1)}])
    with pytest.raises(ValueError, match="position 0 has no logprob"):
        extract_continuation_logprobs(pl, 0, [1])


def test_extract_position_out_of_range_raises():
    pl = make_prompt_logprobs([None, {1: (-1.0, 1)}])
    with pytest.raises(ValueError, match="past the"):
        extract_continuation_logprobs(pl, context_len=1, candidate_token_ids=[1, 2])


def test_extract_missing_token_id_raises():
    pl = make_prompt_logprobs([None, {999: (-1.0, 1)}])
    with pytest.raises(ValueError, match="no prompt logprob for candidate token 1"):
        extract_continuation_logprobs(pl, context_len=1, candidate_token_ids=[1])


# --------------------------------------------------------------------------- #
# make_prompt_logprobs_score_batch_fn
# --------------------------------------------------------------------------- #
class FakeOut:
    def __init__(self, prompt_token_ids, prompt_logprobs):
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs


def test_score_batch_fn_extracts_per_pair():
    # Two pairs: (ctx=[1], cand=[50]) and (ctx=[1,2], cand=[60,61]).
    def fake_generate(sequences, num_prompt_logprobs, context_lens):
        assert num_prompt_logprobs == 0
        assert context_lens == [1, 2]
        # Build per-sequence prompt_logprobs covering the candidate tail.
        assert sequences[0] == [1, 50]
        assert sequences[1] == [1, 2, 60, 61]
        out0 = FakeOut([1, 50], make_prompt_logprobs([None, {50: (-0.7, 1)}]))
        out1 = FakeOut(
            [1, 2, 60, 61],
            make_prompt_logprobs(
                [None, {2: (-0.2, 1)}, {60: (-1.0, 2)}, {61: (-0.3, 1)}]
            ),
        )
        return [out0, out1]

    fn = make_prompt_logprobs_score_batch_fn(fake_generate, num_prompt_logprobs=0)
    results = fn([([1], [50]), ([1, 2], [60, 61])])
    assert results[0] == ([-0.7], [1])
    assert results[1] == ([-1.0, -0.3], [2, 1])


def test_score_batch_fn_output_length_mismatch_raises():
    def bad_generate(sequences, num_prompt_logprobs, context_lens):
        return []

    fn = make_prompt_logprobs_score_batch_fn(bad_generate)
    with pytest.raises(ValueError, match="returned 0 outputs"):
        fn([([1], [50])])
