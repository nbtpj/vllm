# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP API tests for POST /score_choices and POST /rank.

Spins up a real server with a tiny model. Assertions are model-agnostic
(structure, parity, truncation, error handling), not which choice wins.
"""

import pytest
import requests

from vllm.tokenizers import get_tokenizer

from ...utils import RemoteOpenAIServer

MODEL_NAME = "distilbert/distilgpt2"
PROMPT = "The capital of France is"
CHOICES = [" Paris", " London", " a large historic city"]


@pytest.fixture(scope="module")
def server():
    args = ["--max-model-len", "1024", "--enforce-eager"]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def _post(server, path, payload):
    return requests.post(server.url_for(path), json=payload, timeout=60)


def test_score_choices_basic(server):
    r = _post(server, "score_choices", {"prompt": PROMPT, "choices": CHOICES})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["choices"]) == len(CHOICES)
    for ch in body["choices"]:
        assert len(ch["token_logprobs"]) == len(ch["token_ids"])
        assert ch["sum_logprob"] == pytest.approx(sum(ch["token_logprobs"]), abs=1e-3)
    means = [c["mean_logprob"] for c in body["choices"]]
    assert body["best_choice_index"] == means.index(max(means))


def test_score_choices_select_by_sum(server):
    r = _post(
        server,
        "score_choices",
        {"prompt": PROMPT, "choices": CHOICES, "select_by": "sum"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    sums = [c["sum_logprob"] for c in body["choices"]]
    assert body["best_choice_index"] == sums.index(max(sums))


def test_score_choices_text_vs_token_ids_parity(server):
    tok = get_tokenizer(MODEL_NAME)
    ids = [tok.encode(c, add_special_tokens=False) for c in CHOICES]
    a = _post(server, "score_choices", {"prompt": PROMPT, "choices": CHOICES}).json()
    b = _post(server, "score_choices", {"prompt": PROMPT, "choices": ids}).json()
    for ca, cb in zip(a["choices"], b["choices"]):
        assert ca["token_ids"] == cb["token_ids"]
        assert ca["token_logprobs"] == pytest.approx(cb["token_logprobs"], abs=1e-4)


def test_score_choices_empty_choices_is_error(server):
    r = _post(server, "score_choices", {"prompt": PROMPT, "choices": []})
    assert r.status_code == 400


def test_rank_basic(server):
    r = _post(server, "rank", {"prompt": PROMPT, "candidates": CHOICES, "k": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["selected"]) == 2
    assert body["truncated"] is False
    assert [s["order"] for s in body["selected"]] == [0, 1]
    picked = [s["choice_index"] for s in body["selected"]]
    assert len(set(picked)) == len(picked)


def test_rank_truncates_when_k_exceeds_pool(server):
    r = _post(server, "rank", {"prompt": PROMPT, "candidates": CHOICES, "k": 99})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["selected"]) == len(CHOICES)
    assert body["truncated"] is True


def test_rank_k1_matches_score_best(server):
    rank = _post(
        server,
        "rank",
        {"prompt": PROMPT, "candidates": CHOICES, "k": 1, "select_by": "mean"},
    ).json()
    score = _post(
        server,
        "score_choices",
        {"prompt": PROMPT, "choices": CHOICES, "select_by": "mean"},
    ).json()
    assert rank["selected"][0]["choice_index"] == score["best_choice_index"]


def test_rank_negative_k_is_error(server):
    r = _post(server, "rank", {"prompt": PROMPT, "candidates": CHOICES, "k": -1})
    assert r.status_code == 400
