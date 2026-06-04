# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the native fast scoring paths (no model / GPU).

Covers the pure helpers (single-token partitioning, fast-path extraction,
context-priming wave split) and the composed offline scorer driven by a stub
``generate`` that emulates both response shapes (sample-logprobs for the
fast path, prompt-logprobs for the reference path).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.entrypoints.choice_scoring.fast_path import (
    NOT_GREEDY_RANK,
    extract_single_token_results,
    partition_single_token_groups,
    split_priming_wave,
)
from vllm.entrypoints.choice_scoring.offline_api import ChoiceScoringOfflineMixin
from vllm.logprobs import Logprob


# --------------------------------------------------------------------------- #
# partition_single_token_groups
# --------------------------------------------------------------------------- #
def test_partition_groups_by_context():
    pairs = [
        ([1, 2], [10]),
        ([1, 2], [11]),
        ([3], [12]),
    ]
    groups, slow = partition_single_token_groups(pairs)
    assert slow == []
    assert {tuple(g.context): (g.cand_ids, g.pair_indices) for g in groups} == {
        (1, 2): ([10, 11], [0, 1]),
        (3,): ([12], [2]),
    }


def test_partition_multi_token_candidate_makes_whole_group_slow():
    # Mixing fast/slow extraction within one selection is not allowed:
    # one multi-token candidate sends the whole context group to the
    # reference path.
    pairs = [([1], [10]), ([1], [20, 21]), ([2], [12])]
    groups, slow = partition_single_token_groups(pairs)
    assert slow == [0, 1]
    assert len(groups) == 1 and groups[0].context == [2]


def test_partition_rejects_token_id_zero_and_overflow():
    pairs_zero = [([1], [0]), ([1], [10])]
    groups, slow = partition_single_token_groups(pairs_zero)
    assert groups == [] and slow == [0, 1]

    pairs_many = [([1], [10 + i]) for i in range(5)]
    groups, slow = partition_single_token_groups(pairs_many, max_ids=4)
    assert groups == [] and slow == [0, 1, 2, 3, 4]


def test_partition_rejects_empty_context():
    groups, slow = partition_single_token_groups([([], [10])])
    assert groups == [] and slow == [0]


# --------------------------------------------------------------------------- #
# extract_single_token_results
# --------------------------------------------------------------------------- #
def make_group(ctx, cand_ids):
    groups, slow = partition_single_token_groups([(ctx, [c]) for c in cand_ids])
    assert not slow and len(groups) == 1
    return groups[0]


def test_extract_greedy_and_non_greedy():
    group = make_group([1, 2], [10, 11])
    lps = {
        10: Logprob(logprob=-0.5, rank=None),
        11: Logprob(logprob=-2.5, rank=None),
    }
    out = extract_single_token_results(group, 10, lps)
    assert out == [([-0.5], [1]), ([-2.5], [NOT_GREEDY_RANK])]


def test_extract_missing_token_raises():
    group = make_group([1], [10])
    with pytest.raises(ValueError, match="no sample logprob"):
        extract_single_token_results(group, 99, {99: Logprob(logprob=-0.1, rank=1)})
    with pytest.raises(ValueError, match="no sample logprob"):
        extract_single_token_results(group, 99, None)


# --------------------------------------------------------------------------- #
# split_priming_wave
# --------------------------------------------------------------------------- #
def test_priming_splits_shared_long_contexts():
    long_ctx = list(range(100, 140))  # >= 32 tokens
    pairs = [
        (long_ctx, [1, 2]),
        (long_ctx, [3]),
        (long_ctx, [4, 5]),
        ([7], [6]),  # short context: no priming
    ]
    wave1, wave2 = split_priming_wave(pairs)
    assert wave1 == [0, 3]
    assert wave2 == [1, 2]


def test_priming_keeps_singletons_and_short_contexts_in_wave1():
    pairs = [([1] * 40, [2]), ([3] * 5, [4]), ([3] * 5, [5])]
    wave1, wave2 = split_priming_wave(pairs)
    assert wave1 == [0, 1, 2] and wave2 == []


# --------------------------------------------------------------------------- #
# composed offline scorer (stub generate)
# --------------------------------------------------------------------------- #
class StubLLM(ChoiceScoringOfflineMixin):
    """Emulates LLM.generate for both response shapes.

    Sample-logprob requests (logprob_token_ids set) return argmax=token 10
    with fixed per-id logprobs; prompt-logprob requests return -1.0 per
    candidate position with rank 3.
    """

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompts, params, use_tqdm=True, lora_request=None):
        params_list = params if isinstance(params, list) else [params] * len(prompts)
        outs = []
        fast = bool(params_list and params_list[0].logprob_token_ids)
        self.calls.append("fast" if fast else f"ref({len(prompts)})")
        for prompt, sp in zip(prompts, params_list):
            token_ids = list(prompt["prompt_token_ids"])
            if sp.logprob_token_ids:
                lps = {
                    tid: Logprob(logprob=-float(i + 1), rank=None)
                    for i, tid in enumerate(sp.logprob_token_ids)
                }
                sampled = sp.logprob_token_ids[0]
                lps[sampled] = Logprob(logprob=-1.0, rank=1)
                outs.append(
                    SimpleNamespace(
                        prompt_token_ids=token_ids,
                        outputs=[SimpleNamespace(token_ids=[sampled], logprobs=[lps])],
                    )
                )
            else:
                pl = [None] + [
                    {tok: Logprob(logprob=-1.0, rank=3)} for tok in token_ids[1:]
                ]
                outs.append(
                    SimpleNamespace(prompt_token_ids=token_ids, prompt_logprobs=pl)
                )
        return outs


def test_composed_scorer_flag_off_uses_reference(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", raising=False)
    llm = StubLLM()
    fn = llm._make_score_batch_fn(0, False, None)
    out = fn([([1, 2], [10]), ([1, 2], [11])])
    assert llm.calls == ["ref(2)"]
    assert out == [([-1.0], [3]), ([-1.0], [3])]


def test_composed_scorer_fast_path_single_token(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "1")
    llm = StubLLM()
    fn = llm._make_score_batch_fn(0, False, None)
    out = fn([([1, 2], [10]), ([1, 2], [11])])
    # One fused fast call, no reference calls.
    assert llm.calls == ["fast"]
    assert out[0] == ([-1.0], [1])  # argmax candidate
    assert out[1] == ([-2.0], [NOT_GREEDY_RANK])


def test_composed_scorer_mixed_routes_and_priming(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "1")
    llm = StubLLM()
    fn = llm._make_score_batch_fn(0, False, None)
    long_ctx = list(range(1, 41))
    pairs = [
        ([1, 2], [10]),  # fast group
        (long_ctx, [20, 21]),  # slow, priming wave 1
        (long_ctx, [22, 23]),  # slow, priming wave 2
        ([1, 2], [11]),  # fast group
    ]
    out = fn(pairs)
    assert llm.calls == ["fast", "ref(1)", "ref(1)"]
    assert out[0] == ([-1.0], [1])
    assert out[3] == ([-2.0], [NOT_GREEDY_RANK])
    assert out[1] == ([-1.0, -1.0], [3, 3])
    assert out[2] == ([-1.0, -1.0], [3, 3])


def test_composed_scorer_topk_disables_fast_path(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "1")
    llm = StubLLM()
    fn = llm._make_score_batch_fn(2, False, None)
    fn([([1, 2], [10]), ([1, 2], [11])])
    # num_prompt_logprobs > 0 cannot ride the fast path.
    assert all(c.startswith("ref") for c in llm.calls)
