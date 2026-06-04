# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the client glue of engine-resident rank (no model / GPU)."""

from __future__ import annotations

import pytest

from vllm.entrypoints.choice_scoring.native import (
    build_rank_output,
    make_native_rank_params,
)
from vllm.entrypoints.choice_scoring.params import Candidate


def cands(*token_lists):
    return [
        Candidate(token_ids=list(t), index=i, text=f"c{i}")
        for i, t in enumerate(token_lists)
    ]


def test_make_native_rank_params_payload():
    params = make_native_rank_params(cands([1, 2], [3]), k=2, select_by="sum")
    payload = params.extra_args["choice_rank"]
    assert payload == {"candidates": [[1, 2], [3]], "k": 2, "select_by": "sum"}
    assert params.max_tokens == 1
    assert params.temperature == 0.0
    assert params.detokenize is False


def test_build_rank_output_roundtrip():
    pool = cands([1, 2], [3], [4, 5, 6])
    result = {
        "selected": [
            {
                "order": 0,
                "choice_index": 2,
                "token_ids": [4, 5, 6],
                "token_logprobs": [-1.0, -2.0, -3.0],
                "ranks": [1, 2, 3],
            },
            {
                "order": 1,
                "choice_index": 0,
                "token_ids": [1, 2],
                "token_logprobs": [-0.5, -0.5],
                "ranks": [1, 1],
            },
        ],
        "truncated": False,
    }
    out = build_rank_output([9, 9], pool, result)
    assert out.prompt_token_ids == [9, 9]
    assert [s.choice_index for s in out.selected] == [2, 0]
    assert out.selected[0].sum_logprob == pytest.approx(-6.0)
    assert out.selected[0].mean_logprob == pytest.approx(-2.0)
    assert out.selected[1].text == "c0"
    assert out.truncated is False


def test_build_rank_output_error_and_missing():
    pool = cands([1])
    with pytest.raises(ValueError, match="rejected"):
        build_rank_output([9], pool, {"error": "bad payload"})
    with pytest.raises(RuntimeError, match="no choice_rank_result"):
        build_rank_output([9], pool, None)
