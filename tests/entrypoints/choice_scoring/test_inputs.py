# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for input normalization (no model / GPU).

A fake tokenizer maps characters to ids so we can verify text vs token-id vs
dict choices, leading-space handling intent, decoded-token attachment, and
length validation.
"""

from __future__ import annotations

import pytest

from vllm.entrypoints.choice_scoring.inputs import (
    normalize_choices,
    normalize_prompt,
    validate_lengths,
)
from vllm.entrypoints.choice_scoring.params import Candidate


def fake_tokenize(text, add_special_tokens):
    # One id per character (ord); special token -1 prepended when requested.
    ids = [ord(c) for c in text]
    return ([-1] + ids) if add_special_tokens else ids


def fake_decode_tokens(token_ids):
    return [("<bos>" if t == -1 else chr(t)) for t in token_ids]


# --------------------------------------------------------------------------- #
# normalize_prompt
# --------------------------------------------------------------------------- #
def test_normalize_prompt_text_adds_special():
    assert normalize_prompt("ab", fake_tokenize) == [-1, ord("a"), ord("b")]


def test_normalize_prompt_text_no_special():
    assert normalize_prompt("ab", fake_tokenize, add_special_tokens=False) == [
        ord("a"),
        ord("b"),
    ]


def test_normalize_prompt_token_ids_passthrough():
    assert normalize_prompt([5, 6, 7], fake_tokenize) == [5, 6, 7]


def test_normalize_prompt_empty_text_with_bos_is_ok():
    # An empty string still yields a BOS token, which is a valid context.
    assert normalize_prompt("", fake_tokenize) == [-1]


def test_normalize_prompt_empty_text_no_special_raises():
    with pytest.raises(ValueError, match="prompt is empty"):
        normalize_prompt("", fake_tokenize, add_special_tokens=False)


def test_normalize_prompt_empty_ids_raises():
    with pytest.raises(ValueError, match="prompt is empty"):
        normalize_prompt([], fake_tokenize)


# --------------------------------------------------------------------------- #
# normalize_choices
# --------------------------------------------------------------------------- #
def test_choices_text_no_special_tokens():
    cands = normalize_choices(["ab", "c"], fake_tokenize)
    assert cands[0].token_ids == [ord("a"), ord("b")]
    assert cands[1].token_ids == [ord("c")]
    assert cands[0].text == "ab"
    assert [c.index for c in cands] == [0, 1]


def test_choices_leading_space_is_distinct():
    # Intent: a leading space in the choice text produces different tokens.
    with_space = normalize_choices([" a"], fake_tokenize)[0].token_ids
    without = normalize_choices(["a"], fake_tokenize)[0].token_ids
    assert with_space == [ord(" "), ord("a")]
    assert without == [ord("a")]
    assert with_space != without


def test_choices_token_ids_passthrough():
    cands = normalize_choices([[10, 11], [12]], fake_tokenize)
    assert cands[0].token_ids == [10, 11]
    assert cands[1].token_ids == [12]
    assert cands[0].text is None


def test_choices_mixed_text_and_ids():
    cands = normalize_choices(["a", [10, 11]], fake_tokenize)
    assert cands[0].token_ids == [ord("a")]
    assert cands[1].token_ids == [10, 11]


def test_choices_dict_with_token_ids():
    cands = normalize_choices([{"token_ids": [10, 11], "text": "xy"}], fake_tokenize)
    assert cands[0].token_ids == [10, 11]
    assert cands[0].text == "xy"


def test_choices_dict_with_text():
    cands = normalize_choices([{"text": "ab"}], fake_tokenize)
    assert cands[0].token_ids == [ord("a"), ord("b")]


def test_choices_dict_missing_fields_raises():
    with pytest.raises(ValueError, match="must have 'text' or 'token_ids'"):
        normalize_choices([{"foo": 1}], fake_tokenize)


def test_choices_decode_tokens_attached():
    cands = normalize_choices(
        ["ab"], fake_tokenize, decode_tokens_fn=fake_decode_tokens
    )
    assert cands[0].tokens == ["a", "b"]


def test_choices_empty_list_raises():
    with pytest.raises(ValueError, match="at least one"):
        normalize_choices([], fake_tokenize)


def test_choices_empty_text_tokenizes_to_zero_raises():
    with pytest.raises(ValueError, match="zero tokens"):
        normalize_choices([""], fake_tokenize)


def test_choices_empty_token_ids_raises():
    with pytest.raises(ValueError, match="zero tokens"):
        normalize_choices([[]], fake_tokenize)


# --------------------------------------------------------------------------- #
# validate_lengths
# --------------------------------------------------------------------------- #
def _c(ids, i=0):
    return Candidate(token_ids=ids, index=i)


def test_validate_lengths_ok():
    validate_lengths([1, 2], [_c([3, 4], 0)], max_model_len=10)


def test_validate_lengths_none_skips():
    validate_lengths([1, 2], [_c([3] * 100, 0)], max_model_len=None)


def test_validate_lengths_exceeds_raises():
    with pytest.raises(ValueError, match="exceeding max_model_len=5"):
        validate_lengths([1, 2, 3], [_c([4, 5, 6], 0)], max_model_len=5)


def test_validate_lengths_accounts_for_extra_growth():
    # prompt 2 + cand 2 = 4 <= 5, but +2 growth = 6 > 5.
    with pytest.raises(ValueError, match="exceeding max_model_len=5"):
        validate_lengths([1, 2], [_c([3, 4], 0)], max_model_len=5, extra_growth=2)
