# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side scoring & selection for the native (L3) choice-scoring path.

This is the computational heart of the native worker path: given the model's
logits at each candidate position and the actual candidate token ids
(teacher forcing), compute per-token logprobs, then reduce them to per-candidate
``sum``/``mean``/``is_greedy`` and pick the best candidate -- all on-device, with
a ragged (CSR-style) layout so candidates of different lengths are handled
without padding ("no uniform length" requirement).

It deliberately matches the semantics in :mod:`core` (verified by parity tests),
so the native path and the reference oracle agree bit-for-bit up to fp error.
The functions are framework-only (torch) and run on CPU or GPU, which keeps them
unit-testable without CUDA.

Layout
------
``candidate_offsets`` is a CSR row-pointer of shape ``[num_candidates + 1]``;
candidate ``c`` owns flattened positions ``[offsets[c], offsets[c + 1])``.
``logits[p]`` are the next-token logits used to predict the token at flattened
position ``p`` (i.e. conditioned on everything before it), and
``target_token_ids[p]`` is that actual candidate token.
"""

from __future__ import annotations

import torch


def token_logprobs_and_greedy(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced per-position logprob of the target token, and greedy flag.

    Args:
        logits: ``[total_positions, vocab]``.
        target_token_ids: ``[total_positions]`` (the actual candidate tokens).

    Returns:
        ``(token_logprobs[total_positions], is_argmax[total_positions] bool)``
        where ``is_argmax[p]`` is True iff the target token was the argmax.
    """
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    tgt = target_token_ids.long().unsqueeze(1)
    token_logprobs = logprobs.gather(1, tgt).squeeze(1)
    is_argmax = logits.argmax(dim=-1) == target_token_ids
    return token_logprobs, is_argmax


def aggregate_ragged(
    token_logprobs: torch.Tensor,
    is_argmax: torch.Tensor,
    candidate_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce per-token values to per-candidate sum / mean / is_greedy.

    Args:
        token_logprobs: ``[total_positions]``.
        is_argmax: ``[total_positions]`` bool.
        candidate_offsets: ``[num_candidates + 1]`` CSR row pointer.

    Returns:
        ``(sum_logprob[C], mean_logprob[C], is_greedy[C] bool)``.
    """
    num_candidates = candidate_offsets.numel() - 1
    lengths = (candidate_offsets[1:] - candidate_offsets[:-1]).to(token_logprobs.device)
    if (lengths <= 0).any():
        raise ValueError("every candidate must have at least one token")

    seg_ids = torch.repeat_interleave(
        torch.arange(num_candidates, device=token_logprobs.device), lengths
    )
    sum_logprob = torch.zeros(num_candidates, device=token_logprobs.device).index_add_(
        0, seg_ids, token_logprobs
    )
    mean_logprob = sum_logprob / lengths.to(sum_logprob.dtype)

    greedy_count = torch.zeros(num_candidates, device=token_logprobs.device).index_add_(
        0, seg_ids, is_argmax.to(sum_logprob.dtype)
    )
    is_greedy = greedy_count == lengths.to(sum_logprob.dtype)
    return sum_logprob, mean_logprob, is_greedy


def select_best(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> int:
    """Argmax over ``scores`` with lowest-index tie-breaking.

    ``valid_mask`` (bool, same shape) excludes already-selected candidates from
    the running pool during ranking. Returns ``-1`` if nothing is valid.
    """
    masked = scores
    if valid_mask is not None:
        masked = scores.masked_fill(~valid_mask, float("-inf"))
    if torch.isinf(masked).all() and (masked == float("-inf")).all():
        return -1
    best = masked.max()
    if best == float("-inf"):
        return -1
    # Lowest index achieving the max (mirrors core.select_best_index).
    winners = (masked == best).nonzero(as_tuple=False).flatten()
    return int(winners.min().item())


def score_candidates(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    candidate_offsets: torch.Tensor,
    select_by: str = "mean",
) -> dict[str, torch.Tensor]:
    """End-to-end native scoring for one prompt's candidate batch.

    Returns a dict with ``token_logprobs`` (ragged, ``[total_positions]``),
    ``sum_logprob`` / ``mean_logprob`` / ``is_greedy`` (``[C]``), and
    ``best_index`` (python int) under the chosen ``select_by`` metric.
    """
    token_logprobs, is_argmax = token_logprobs_and_greedy(logits, target_token_ids)
    sum_lp, mean_lp, is_greedy = aggregate_ragged(
        token_logprobs, is_argmax, candidate_offsets
    )
    if select_by == "mean":
        metric = mean_lp
    elif select_by == "sum":
        metric = sum_lp
    else:
        raise ValueError(f"unknown select_by={select_by!r}")
    return {
        "token_logprobs": token_logprobs,
        "sum_logprob": sum_lp,
        "mean_logprob": mean_lp,
        "is_greedy": is_greedy,
        "best_index": select_best(metric),
    }
