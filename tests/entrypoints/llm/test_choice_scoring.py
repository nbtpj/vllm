# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end offline tests for LLM.score_choices() and LLM.rank().

Requires a working accelerator + a tiny causal LM (distilgpt2). Assertions are
model-agnostic: they check internal consistency, text/token-id parity, the
top-k invariance of the chosen-token logprob, the rank<->score cross-parity,
determinism, and edge/truncation behavior -- not which specific choice "wins"
(distilgpt2 is too weak to rely on that).
"""

from __future__ import annotations

import math
import weakref

import pytest

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "distilbert/distilgpt2"


@pytest.fixture(scope="module")
def llm():
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.10,
        enforce_eager=True,
    )
    yield weakref.proxy(llm)
    del llm
    cleanup_dist_env_and_memory()


PROMPT = "The capital of France is"
CHOICES = [" Paris", " London", " a very large and historic city"]


# --------------------------------------------------------------------------- #
# score_choices
# --------------------------------------------------------------------------- #
@pytest.mark.skip_global_cleanup
def test_score_choices_basic_consistency(llm: LLM):
    out = llm.batch_score(PROMPT, CHOICES)
    assert len(out.choices) == len(CHOICES)
    for ch in out.choices:
        assert len(ch.token_logprobs) == len(ch.token_ids)
        assert ch.sum_logprob == pytest.approx(sum(ch.token_logprobs), abs=1e-4)
        assert ch.mean_logprob == pytest.approx(
            ch.sum_logprob / len(ch.token_ids), abs=1e-5
        )
        assert isinstance(ch.is_greedy, bool)
    # best_choice_index is the argmax by mean (default), lowest-index tie-break.
    means = [c.mean_logprob for c in out.choices]
    assert out.best_choice_index == means.index(max(means))


@pytest.mark.skip_global_cleanup
def test_score_choices_select_by_sum(llm: LLM):
    out = llm.batch_score(PROMPT, CHOICES, select_by="sum")
    sums = [c.sum_logprob for c in out.choices]
    assert out.best_choice_index == sums.index(max(sums))


@pytest.mark.skip_global_cleanup
def test_score_choices_text_vs_token_ids_parity(llm: LLM):
    tok = llm.get_tokenizer()
    choice_ids = [tok.encode(c, add_special_tokens=False) for c in CHOICES]
    by_text = llm.batch_score(PROMPT, CHOICES)
    by_ids = llm.batch_score(PROMPT, choice_ids)
    for a, b in zip(by_text.choices, by_ids.choices):
        assert a.token_ids == b.token_ids
        assert a.token_logprobs == pytest.approx(b.token_logprobs, abs=1e-5)


@pytest.mark.skip_global_cleanup
def test_score_choices_chosen_logprob_topk_invariant(llm: LLM):
    # The actual token's logprob must not depend on how many top-k we request.
    a = llm.batch_score(PROMPT, CHOICES, num_prompt_logprobs=0)
    b = llm.batch_score(PROMPT, CHOICES, num_prompt_logprobs=5)
    for ca, cb in zip(a.choices, b.choices):
        assert ca.token_logprobs == pytest.approx(cb.token_logprobs, abs=1e-5)


@pytest.mark.skip_global_cleanup
def test_score_choices_logprobs_are_valid(llm: LLM):
    out = llm.batch_score(PROMPT, CHOICES)
    for ch in out.choices:
        for lp in ch.token_logprobs:
            assert lp <= 1e-4  # log-prob of a probability is <= 0
            assert not math.isnan(lp)


@pytest.mark.skip_global_cleanup
def test_score_choices_batch_independent(llm: LLM):
    prompts = ["The capital of France is", "Two plus two equals"]
    choices = [[" Paris", " London"], [" four", " five", " purple"]]
    outs = llm.batch_score(prompts, choices)
    assert isinstance(outs, list) and len(outs) == 2
    assert len(outs[0].choices) == 2
    assert len(outs[1].choices) == 3


@pytest.mark.skip_global_cleanup
def test_score_choices_deterministic(llm: LLM):
    a = llm.batch_score(PROMPT, CHOICES)
    b = llm.batch_score(PROMPT, CHOICES)
    assert a.best_choice_index == b.best_choice_index
    for ca, cb in zip(a.choices, b.choices):
        assert ca.token_logprobs == pytest.approx(cb.token_logprobs, abs=1e-6)


# --------------------------------------------------------------------------- #
# rank
# --------------------------------------------------------------------------- #
@pytest.mark.skip_global_cleanup
def test_rank_basic_shape(llm: LLM):
    out = llm.batch_rank(PROMPT, CHOICES, k=2)
    assert len(out.selected) == 2
    assert out.truncated is False
    assert [s.order for s in out.selected] == [0, 1]
    picked = [s.choice_index for s in out.selected]
    assert len(set(picked)) == len(picked)
    assert all(0 <= i < len(CHOICES) for i in picked)


@pytest.mark.skip_global_cleanup
def test_rank_full_ordering_is_permutation(llm: LLM):
    out = llm.batch_rank(PROMPT, CHOICES, k=len(CHOICES))
    picked = sorted(s.choice_index for s in out.selected)
    assert picked == list(range(len(CHOICES)))
    assert out.truncated is False


@pytest.mark.skip_global_cleanup
def test_rank_k_exceeds_pool_truncates(llm: LLM):
    out = llm.batch_rank(PROMPT, CHOICES, k=99)
    assert len(out.selected) == len(CHOICES)
    assert out.truncated is True


@pytest.mark.skip_global_cleanup
def test_rank_k1_matches_score_best(llm: LLM):
    # rank step 0 scores against the prompt alone -> identical to score_choices.
    ranked = llm.batch_rank(PROMPT, CHOICES, k=1, select_by="mean")
    scored = llm.batch_score(PROMPT, CHOICES, select_by="mean")
    assert ranked.selected[0].choice_index == scored.best_choice_index


@pytest.mark.skip_global_cleanup
def test_rank_deterministic(llm: LLM):
    a = llm.batch_rank(PROMPT, CHOICES, k=len(CHOICES))
    b = llm.batch_rank(PROMPT, CHOICES, k=len(CHOICES))
    assert [s.choice_index for s in a.selected] == [s.choice_index for s in b.selected]


@pytest.mark.skip_global_cleanup
def test_rank_per_token_logprobs_recorded(llm: LLM):
    out = llm.batch_rank(PROMPT, CHOICES, k=2)
    for step in out.selected:
        assert len(step.token_logprobs) == len(step.token_ids)
        assert step.mean_logprob == pytest.approx(
            step.sum_logprob / len(step.token_ids), abs=1e-5
        )


@pytest.mark.skip_global_cleanup
def test_rank_batch_per_prompt_k(llm: LLM):
    prompts = ["The capital of France is", "Two plus two equals"]
    cands = [[" Paris", " London", " Berlin"], [" four", " five"]]
    outs = llm.batch_rank(prompts, cands, k=[2, 1])
    assert len(outs) == 2
    assert len(outs[0].selected) == 2
    assert len(outs[1].selected) == 1


# --------------------------------------------------------------------------- #
# native window parity (VLLM_ENABLE_NATIVE_CHOICE_SCORING)
# --------------------------------------------------------------------------- #
# The reference path computes prompt logprobs at every position; the native
# window restricts the LM head to the candidate positions. Same inputs must
# give the same outputs within fp tolerance -- the reference is the oracle.
@pytest.mark.skip_global_cleanup
def test_native_window_score_parity(llm: LLM, monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", raising=False)
    base = llm.batch_score(PROMPT, CHOICES)
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "1")
    native = llm.batch_score(PROMPT, CHOICES)
    assert native.best_choice_index == base.best_choice_index
    for cb, cn in zip(base.choices, native.choices):
        assert cn.token_ids == cb.token_ids
        assert cn.token_logprobs == pytest.approx(cb.token_logprobs, abs=1e-3)
        assert cn.is_greedy == cb.is_greedy


@pytest.mark.skip_global_cleanup
def test_native_window_score_parity_with_topk(llm: LLM, monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", raising=False)
    base = llm.batch_score(PROMPT, CHOICES, num_prompt_logprobs=5)
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "1")
    native = llm.batch_score(PROMPT, CHOICES, num_prompt_logprobs=5)
    for cb, cn in zip(base.choices, native.choices):
        assert cn.token_logprobs == pytest.approx(cb.token_logprobs, abs=1e-3)


@pytest.mark.skip_global_cleanup
def test_native_window_rank_parity(llm: LLM, monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", raising=False)
    base = llm.batch_rank(PROMPT, CHOICES, k=len(CHOICES))
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "1")
    native = llm.batch_rank(PROMPT, CHOICES, k=len(CHOICES))
    assert [s.choice_index for s in native.selected] == [
        s.choice_index for s in base.selected
    ]
    assert native.truncated == base.truncated
    for sb, sn in zip(base.selected, native.selected):
        assert sn.token_logprobs == pytest.approx(sb.token_logprobs, abs=1e-3)


@pytest.mark.skip_global_cleanup
def test_native_window_batch_parity(llm: LLM, monkeypatch):
    prompts = ["The capital of France is", "Two plus two equals"]
    choices = [[" Paris", " London"], [" four", " five", " purple"]]
    monkeypatch.delenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", raising=False)
    base = llm.batch_score(prompts, choices)
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "1")
    native = llm.batch_score(prompts, choices)
    for ob, on in zip(base, native):
        assert on.best_choice_index == ob.best_choice_index
        for cb, cn in zip(ob.choices, on.choices):
            assert cn.token_logprobs == pytest.approx(cb.token_logprobs, abs=1e-3)
