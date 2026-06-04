# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the native prompt-logprobs window (no model / GPU).

Covers the CPU-visible pieces of the native choice-scoring path:

* ``SamplingParams.prompt_logprobs_from`` validation,
* ``make_scoring_sampling_params`` honoring VLLM_ENABLE_NATIVE_CHOICE_SCORING,
* ``LogprobsProcessor`` None-padding positions below the window so received
  rows land at their absolute prompt positions.

The GPU halves (both model runners computing only windowed rows) are covered
by the parity smoke on real hardware: flag on vs off must give identical
batch_score / batch_rank outputs.
"""

from __future__ import annotations

import pytest

from vllm.entrypoints.choice_scoring.reference import (
    make_scoring_sampling_params,
)
from vllm.sampling_params import SamplingParams


# --------------------------------------------------------------------------- #
# SamplingParams validation
# --------------------------------------------------------------------------- #
def test_prompt_logprobs_from_requires_prompt_logprobs():
    with pytest.raises(Exception, match="requires prompt_logprobs"):
        SamplingParams(prompt_logprobs_from=3)


def test_prompt_logprobs_from_must_be_positive():
    with pytest.raises(Exception, match="must be >= 1"):
        SamplingParams(prompt_logprobs=0, prompt_logprobs_from=0)


def test_prompt_logprobs_from_valid():
    sp = SamplingParams(prompt_logprobs=0, prompt_logprobs_from=4)
    assert sp.prompt_logprobs_from == 4


# --------------------------------------------------------------------------- #
# make_scoring_sampling_params x VLLM_ENABLE_NATIVE_CHOICE_SCORING
# --------------------------------------------------------------------------- #
def test_scoring_params_default_no_window(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", raising=False)
    sp = make_scoring_sampling_params(0, context_len=7)
    assert sp.prompt_logprobs == 0
    assert sp.prompt_logprobs_from is None
    assert sp.max_tokens == 1
    assert sp.temperature == 0.0
    assert sp.detokenize is False


def test_scoring_params_window_when_flag_on(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "1")
    sp = make_scoring_sampling_params(0, context_len=7)
    assert sp.prompt_logprobs_from == 7


def test_scoring_params_no_window_without_context_len(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "1")
    sp = make_scoring_sampling_params(0, context_len=None)
    assert sp.prompt_logprobs_from is None


def test_scoring_params_flag_off_explicit(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_NATIVE_CHOICE_SCORING", "0")
    sp = make_scoring_sampling_params(2, context_len=5)
    assert sp.prompt_logprobs == 2
    assert sp.prompt_logprobs_from is None


# --------------------------------------------------------------------------- #
# LogprobsProcessor None-padding below the window
# --------------------------------------------------------------------------- #
class _FakeRequest:
    def __init__(self, sampling_params):
        self.sampling_params = sampling_params


def _make_processor(sampling_params):
    from vllm.v1.engine.logprobs import LogprobsProcessor

    return LogprobsProcessor.from_new_request(None, _FakeRequest(sampling_params))


def test_logprobs_processor_pads_nones_below_window():
    proc = _make_processor(SamplingParams(prompt_logprobs=0, prompt_logprobs_from=4))
    # Positions 0..3 are None (position 0 always is; 1..3 are below the
    # window); received rows then land at positions 4+.
    assert list(proc.prompt_logprobs) == [None, None, None, None]


def test_logprobs_processor_no_window_single_none():
    proc = _make_processor(SamplingParams(prompt_logprobs=0))
    assert list(proc.prompt_logprobs) == [None]


def test_logprobs_processor_window_rows_land_at_absolute_positions():
    import torch

    from vllm.v1.outputs import LogprobsTensors

    proc = _make_processor(SamplingParams(prompt_logprobs=0, prompt_logprobs_from=3))
    # Two windowed rows: prompt token positions 3 and 4.
    tensors = LogprobsTensors(
        logprob_token_ids=torch.tensor([[30], [40]]),
        logprobs=torch.tensor([[-0.5], [-1.5]]),
        selected_token_ranks=torch.tensor([1, 2]),
    )
    proc._update_prompt_logprobs(tensors)
    plp = proc.prompt_logprobs
    assert len(plp) == 5
    assert plp[0] is None and plp[1] is None and plp[2] is None
    assert plp[3][30].logprob == pytest.approx(-0.5)
    assert plp[3][30].rank == 1
    assert plp[4][40].logprob == pytest.approx(-1.5)
    assert plp[4][40].rank == 2
