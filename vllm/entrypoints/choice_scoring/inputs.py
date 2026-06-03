# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input normalization for choice scoring & ranking.

Prompts and choices may be supplied as text or as pre-tokenized id lists (or,
for the HTTP layer, as small dicts). This module turns them into
:class:`Candidate` objects with token ids (and optional decoded tokens), and
validates lengths. Tokenizer/detokenizer access is injected so the logic is
unit-testable without loading a model.

Note on tokenization of choices: when a choice is given as text, whether a
leading space belongs to the choice (``" Paris"`` vs ``"Paris"``) is
significant for many tokenizers. Callers should include the leading space in
the choice text when that is the intended continuation. Special tokens are
**not** added to choices by default (``add_special_tokens=False``); they are
continuations, not standalone sequences.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from vllm.entrypoints.choice_scoring.params import Candidate

# tokenize_fn(text, add_special_tokens) -> token ids
TokenizeFn = Callable[[str, bool], list[int]]
# decode_fn(token_ids) -> list of per-token strings (one per id)
DecodeTokensFn = Callable[[Sequence[int]], list[str]]

# A raw choice: text, a token-id list, or a dict with "text"/"token_ids".
RawChoice = str | Sequence[int] | dict[str, Any]
# A raw prompt: text or a token-id list.
RawPrompt = str | Sequence[int]


def normalize_prompt(
    prompt: RawPrompt,
    tokenize_fn: TokenizeFn,
    add_special_tokens: bool = True,
) -> list[int]:
    """Tokenize a prompt to ids (text gets special tokens, e.g. BOS, by default)."""
    if isinstance(prompt, str):
        ids = tokenize_fn(prompt, add_special_tokens)
    else:
        ids = list(prompt)
    if len(ids) == 0:
        raise ValueError(
            "prompt is empty; the context must contain at least one token (the "
            "first continuation token needs something to condition on)"
        )
    return ids


def _choice_to_token_ids(
    choice: RawChoice,
    tokenize_fn: TokenizeFn,
) -> tuple[list[int], str | None]:
    """Return ``(token_ids, text_or_None)`` for one raw choice."""
    if isinstance(choice, str):
        return tokenize_fn(choice, False), choice
    if isinstance(choice, dict):
        if "token_ids" in choice and choice["token_ids"] is not None:
            return list(choice["token_ids"]), choice.get("text")
        if "text" in choice and choice["text"] is not None:
            return tokenize_fn(choice["text"], False), choice["text"]
        raise ValueError("choice dict must have 'text' or 'token_ids'")
    # Assume a sequence of ints.
    return list(choice), None


def normalize_choices(
    choices: Sequence[RawChoice],
    tokenize_fn: TokenizeFn,
    decode_tokens_fn: DecodeTokensFn | None = None,
) -> list[Candidate]:
    """Turn raw choices into validated :class:`Candidate` objects.

    Each candidate gets a stable ``index`` (its position in ``choices``). If
    ``decode_tokens_fn`` is given, per-token strings are attached for display.

    Raises:
        ValueError: if there are no choices or any choice tokenizes to empty.
    """
    if len(choices) == 0:
        raise ValueError("at least one choice/candidate is required")

    candidates: list[Candidate] = []
    for i, raw in enumerate(choices):
        token_ids, text = _choice_to_token_ids(raw, tokenize_fn)
        if len(token_ids) == 0:
            raise ValueError(
                f"choice at position {i} ({raw!r}) tokenized to zero tokens"
            )
        tokens = decode_tokens_fn(token_ids) if decode_tokens_fn else None
        candidates.append(
            Candidate(token_ids=token_ids, text=text, tokens=tokens, index=i)
        )
    return candidates


def validate_lengths(
    prompt_token_ids: Sequence[int],
    candidates: Sequence[Candidate],
    max_model_len: int | None,
    extra_growth: int = 0,
) -> None:
    """Ensure ``prompt + candidate (+ extra_growth)`` fits in ``max_model_len``.

    ``extra_growth`` accounts for tokens that will be appended before/around a
    candidate during ranking (e.g. previously selected bundles). Pass the worst
    case so the longest sequence we will ever build is validated up front.
    """
    if max_model_len is None:
        return
    for cand in candidates:
        total = len(prompt_token_ids) + len(cand.token_ids) + extra_growth
        if total > max_model_len:
            raise ValueError(
                f"choice {cand.index} would build a sequence of {total} tokens, "
                f"exceeding max_model_len={max_model_len}"
            )
