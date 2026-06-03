# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the native (L3) device-side scoring ops, on CPU torch.

These prove the on-device tensor path produces the same per-token logprobs,
sum/mean/greedy and best-choice selection as the pure-Python :mod:`core`
reference -- the contract the GPU native path must uphold.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm.entrypoints.choice_scoring.core import (  # noqa: E402
    aggregate_choice,
    select_best_index,
)
from vllm.entrypoints.choice_scoring.native_tensor_ops import (  # noqa: E402
    aggregate_ragged,
    score_candidates,
    select_best,
    token_logprobs_and_greedy,
)
from vllm.entrypoints.choice_scoring.params import Candidate  # noqa: E402


def _ref_logprobs(logits, target_ids):
    """Pure-torch reference (independent of model_runner) for cross-check."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    return [lp[i, t].item() for i, t in enumerate(target_ids)]


def test_token_logprobs_match_manual_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    targets = torch.tensor([3, 1, 9, 0])
    tlp, is_argmax = token_logprobs_and_greedy(logits, targets)
    expected = _ref_logprobs(logits, targets.tolist())
    assert tlp.tolist() == pytest.approx(expected, abs=1e-5)
    # greedy flag matches argmax.
    assert is_argmax.tolist() == (logits.argmax(-1) == targets).tolist()


def test_aggregate_ragged_matches_core():
    # Two candidates of lengths 2 and 3 (no uniform length).
    token_logprobs = torch.tensor([-0.5, -1.5, -0.1, -0.2, -0.3])
    is_argmax = torch.tensor([True, True, True, False, True])
    offsets = torch.tensor([0, 2, 5])
    sum_lp, mean_lp, is_greedy = aggregate_ragged(token_logprobs, is_argmax, offsets)

    c0 = aggregate_choice(Candidate([1, 2], index=0), [-0.5, -1.5], ranks=[1, 1])
    c1 = aggregate_choice(
        Candidate([3, 4, 5], index=1), [-0.1, -0.2, -0.3], ranks=[1, 4, 1]
    )
    assert sum_lp.tolist() == pytest.approx([c0.sum_logprob, c1.sum_logprob], abs=1e-6)
    assert mean_lp.tolist() == pytest.approx(
        [c0.mean_logprob, c1.mean_logprob], abs=1e-6
    )
    assert is_greedy.tolist() == [c0.is_greedy, c1.is_greedy] == [True, False]


def test_aggregate_ragged_rejects_empty_candidate():
    with pytest.raises(ValueError, match="at least one token"):
        aggregate_ragged(
            torch.tensor([-0.5]), torch.tensor([True]), torch.tensor([0, 0, 1])
        )


def test_select_best_matches_core_tie_break():
    # All tied -> lowest index, mirroring core.select_best_index.
    scores = torch.tensor([-1.0, -1.0, -1.0])
    assert select_best(scores) == 0
    ref = [aggregate_choice(Candidate([i], index=i), [-1.0]) for i in range(3)]
    assert select_best_index(ref, "mean") == 0


def test_select_best_respects_valid_mask():
    scores = torch.tensor([-0.1, -0.5, -0.2])  # index 0 best overall
    mask = torch.tensor([False, True, True])  # but 0 already selected
    assert select_best(scores, mask) == 2  # -0.2 > -0.5


def test_select_best_all_invalid_returns_minus_one():
    scores = torch.tensor([-0.1, -0.2])
    assert select_best(scores, torch.tensor([False, False])) == -1


def test_score_candidates_end_to_end_matches_core():
    torch.manual_seed(1)
    vocab = 16
    # candidate 0 = [2,5], candidate 1 = [7]
    logits = torch.randn(3, vocab)
    targets = torch.tensor([2, 5, 7])
    offsets = torch.tensor([0, 2, 3])

    out = score_candidates(logits, targets, offsets, select_by="mean")

    # Build the core reference from the same logprobs.
    tlp = _ref_logprobs(logits, targets.tolist())
    c0 = aggregate_choice(Candidate([2, 5], index=0), tlp[0:2])
    c1 = aggregate_choice(Candidate([7], index=1), tlp[2:3])
    assert out["sum_logprob"].tolist() == pytest.approx(
        [c0.sum_logprob, c1.sum_logprob], abs=1e-5
    )
    assert out["mean_logprob"].tolist() == pytest.approx(
        [c0.mean_logprob, c1.mean_logprob], abs=1e-5
    )
    assert out["best_index"] == select_best_index([c0, c1], "mean")


def test_score_candidates_select_by_sum_vs_mean():
    # candidate 0 len2 sum -0.2 mean -0.1 ; candidate 1 len1 sum/mean -0.15
    # Construct logits so the gathered logprobs hit those values approximately
    # by using a 1-hot-ish large vocab; instead just check the metric routing.
    vocab = 8
    logits = torch.zeros(3, vocab)
    # make target logprobs deterministic via log_softmax of crafted logits
    # easier: directly test routing by monkey-checking both metrics differ.
    targets = torch.tensor([0, 0, 0])
    offsets = torch.tensor([0, 2, 3])
    by_mean = score_candidates(logits, targets, offsets, select_by="mean")
    by_sum = score_candidates(logits, targets, offsets, select_by="sum")
    # equal logits -> equal per-token logprob; len2 has larger (more negative)
    # sum but equal mean -> sum prefers the shorter candidate (index 1), mean ties
    # to index 0.
    assert by_sum["best_index"] == 1
    assert by_mean["best_index"] == 0


def test_score_candidates_unknown_select_by_raises():
    with pytest.raises(ValueError, match="unknown select_by"):
        score_candidates(
            torch.randn(1, 4), torch.tensor([0]), torch.tensor([0, 1]), select_by="x"
        )
