# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure helpers for the native fast scoring paths (no engine dependency).

Two optimizations composed by the offline/serving scorers when
``VLLM_ENABLE_NATIVE_CHOICE_SCORING`` is on:

* **Single-token fast path** -- a context whose candidates are all single
  tokens needs no teacher forcing at all: one request on the bare context
  with ``SamplingParams.logprob_token_ids=[cand ids]`` returns every
  candidate's logprob from a single forward (and the argmax token, since
  ``temperature=0``). N choices collapse from N sequences to 1 request.
* **Context-priming wave** -- pairs submitted in the same engine wave cannot
  reuse each other's prefix-cache blocks (blocks only become readable after
  they are computed). Splitting a multi-pair context group into wave 1 (one
  pair, computes + caches the context) and wave 2 (the rest, hits the cache)
  makes the shared-context prefill happen once instead of N times.

All functions here are pure and CPU-unit-tested; the generate calls live in
the offline/serving wrappers.
"""

from __future__ import annotations

from typing import NamedTuple

from vllm.entrypoints.choice_scoring.core import ScoredContinuation
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS

Pair = tuple[list[int], list[int]]

# Sentinel vocab rank for "not the argmax": the exact rank is unknown on the
# fast path (only the sampled token's rank is real), but ranks are never
# exposed in outputs -- they only feed ``is_greedy = all(rank == 1)``.
NOT_GREEDY_RANK = 2


class FastGroup(NamedTuple):
    context: list[int]
    cand_ids: list[int]  # one single-token candidate id per pair
    pair_indices: list[int]  # positions in the original pairs list


def partition_single_token_groups(
    pairs: list[Pair],
    max_ids: int = MAX_LOGPROB_TOKEN_IDS,
) -> tuple[list[FastGroup], list[int]]:
    """Split pairs into fast single-token context groups and the rest.

    A context group rides the fast path only when *every* candidate of that
    context is a single token (mixing extraction paths within one selection
    would compare numerically-different scores), the group fits the
    ``logprob_token_ids`` length cap, and no candidate is token id 0 (the
    sampler pads heterogeneous id lists with 0/-inf entries that could
    shadow a real id-0 logprob).

    Returns ``(fast_groups, slow_indices)``; every original pair index
    appears in exactly one of the two.
    """
    by_ctx: dict[tuple[int, ...], list[int]] = {}
    for i, (ctx, _) in enumerate(pairs):
        by_ctx.setdefault(tuple(ctx), []).append(i)

    fast_groups: list[FastGroup] = []
    slow_indices: list[int] = []
    for ctx_key, indices in by_ctx.items():
        cands = [pairs[i][1] for i in indices]
        eligible = (
            len(ctx_key) > 0
            and len(indices) <= max_ids
            and all(len(c) == 1 for c in cands)
            and all(c[0] != 0 for c in cands)
        )
        if eligible:
            fast_groups.append(
                FastGroup(
                    context=list(ctx_key),
                    cand_ids=[c[0] for c in cands],
                    pair_indices=list(indices),
                )
            )
        else:
            slow_indices.extend(indices)
    slow_indices.sort()
    return fast_groups, slow_indices


def extract_single_token_results(
    group: FastGroup,
    sampled_token_id: int,
    position_logprobs: dict,
) -> list[ScoredContinuation]:
    """Convert one fast-path response into per-pair results (group order).

    ``position_logprobs`` is the sample-logprobs dict of the first generated
    position ({token_id: Logprob}); ``sampled_token_id`` is the argmax token
    (``temperature=0``), which determines ``is_greedy`` exactly.
    """
    results: list[ScoredContinuation] = []
    for cand_id in group.cand_ids:
        if position_logprobs is None or cand_id not in position_logprobs:
            raise ValueError(
                f"no sample logprob returned for requested token {cand_id}"
            )
        logprob = float(position_logprobs[cand_id].logprob)
        rank = 1 if cand_id == sampled_token_id else NOT_GREEDY_RANK
        results.append(([logprob], [rank]))
    return results


def split_priming_wave(
    pairs: list[Pair],
    min_context_len: int = 32,
) -> tuple[list[int], list[int]]:
    """Pick wave-1 (context-priming) and wave-2 indices for slow pairs.

    For each context shared by >= 2 pairs and long enough to span at least
    one cache block, the first pair goes to wave 1 and the siblings to
    wave 2 (where they hit the context blocks wave 1 cached). Everything
    else goes to wave 1. Returns ``(wave1, wave2)`` index lists.
    """
    by_ctx: dict[tuple[int, ...], list[int]] = {}
    for i, (ctx, _) in enumerate(pairs):
        by_ctx.setdefault(tuple(ctx), []).append(i)

    wave1: list[int] = []
    wave2: list[int] = []
    for ctx_key, indices in by_ctx.items():
        if len(indices) >= 2 and len(ctx_key) >= min_context_len:
            wave1.append(indices[0])
            wave2.extend(indices[1:])
        else:
            wave1.extend(indices)
    wave1.sort()
    wave2.sort()
    return wave1, wave2
