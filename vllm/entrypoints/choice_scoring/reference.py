# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference (oracle) scorer built on the existing ``prompt_logprobs`` path.

This implements a :data:`~vllm.entrypoints.choice_scoring.batching.ScoreBatchFn`
by teacher forcing: for each ``(context, candidate)`` pair we run the normal
generate path over the concatenation ``context + candidate`` requesting
``prompt_logprobs`` and read off the actual candidate tokens' logprobs (and
vocab ranks) at the candidate positions. Prefix caching makes the shared
context essentially free across the choices of a prompt.

It is intentionally simple and obviously correct, so it doubles as:
* the **fallback** path (works on any causal LM, no native kernels), and
* the **oracle** the native L3 path is parity-tested against.

The numeric extraction (:func:`extract_continuation_logprobs`) is a pure
function and is unit-tested directly with synthetic logprobs; the
``generate_fn`` is injected so this module has no hard dependency on ``LLM``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from vllm.entrypoints.choice_scoring.core import ScoredContinuation
from vllm.logprobs import Logprob, PromptLogprobs

# A callable that runs the generate path over a batch of token-id sequences and
# returns one object per sequence exposing ``.prompt_token_ids`` and
# ``.prompt_logprobs`` (i.e. a vLLM ``RequestOutput``). Order must be preserved.
GenerateFn = Callable[[list[list[int]], int], list["_HasPromptLogprobs"]]


class _HasPromptLogprobs:
    """Structural type marker (for readability only)."""

    prompt_token_ids: list[int]
    prompt_logprobs: PromptLogprobs | None


def extract_continuation_logprobs(
    prompt_logprobs: PromptLogprobs | None,
    context_len: int,
    candidate_token_ids: Sequence[int],
) -> ScoredContinuation:
    """Read the teacher-forced logprobs/ranks of a candidate continuation.

    ``prompt_logprobs[p]`` holds the logprobs for the token at prompt position
    ``p`` conditioned on positions ``< p`` (position 0 is ``None``). The
    candidate occupies prompt positions ``[context_len, context_len + L)``; the
    j-th candidate token's logprob is the entry keyed by that token id at
    position ``context_len + j``.

    Returns ``(token_logprobs, ranks)``. ``ranks`` is ``None`` if any position
    lacks a vocab rank; otherwise a 1-based rank per token (rank 1 == argmax).

    Raises:
        ValueError: if ``prompt_logprobs`` is missing, the candidate is empty,
            positions run past the returned logprobs, or a candidate token's
            logprob is absent (e.g. a position-0 candidate with no context).
    """
    n = len(candidate_token_ids)
    if n == 0:
        raise ValueError("candidate has zero tokens")
    if prompt_logprobs is None:
        raise ValueError(
            "prompt_logprobs is None; request must set prompt_logprobs and the "
            "model/backend must support it"
        )

    logprobs: list[float] = []
    ranks: list[int] = []
    have_all_ranks = True
    for j, token_id in enumerate(candidate_token_ids):
        pos = context_len + j
        if pos == 0:
            raise ValueError(
                "candidate token at absolute position 0 has no logprob; the "
                "context must be non-empty (e.g. include BOS)"
            )
        if pos >= len(prompt_logprobs):
            raise ValueError(
                f"position {pos} is past the {len(prompt_logprobs)} returned "
                "prompt_logprobs"
            )
        position_logprobs = prompt_logprobs[pos]
        if position_logprobs is None or token_id not in position_logprobs:
            raise ValueError(
                f"no prompt logprob for candidate token {token_id} at position "
                f"{pos}; consider increasing the requested prompt_logprobs count"
            )
        entry: Logprob = position_logprobs[token_id]
        logprobs.append(float(entry.logprob))
        if entry.rank is None:
            have_all_ranks = False
        else:
            ranks.append(int(entry.rank))

    return logprobs, (ranks if have_all_ranks and len(ranks) == n else None)


def make_prompt_logprobs_score_batch_fn(
    generate_fn: GenerateFn,
    num_prompt_logprobs: int = 0,
):
    """Build a ``ScoreBatchFn`` backed by ``generate_fn`` + ``prompt_logprobs``.

    ``generate_fn(token_id_sequences, num_prompt_logprobs)`` must run the
    generate path (``max_tokens=1``, ``temperature=0``, ``prompt_logprobs`` set)
    and return outputs in input order.
    """

    def _score_batch(
        pairs: list[tuple[list[int], list[int]]],
    ) -> list[ScoredContinuation]:
        sequences = [list(ctx) + list(cand) for ctx, cand in pairs]
        context_lens = [len(ctx) for ctx, _ in pairs]
        candidates = [cand for _, cand in pairs]

        outputs = generate_fn(sequences, num_prompt_logprobs)
        if len(outputs) != len(pairs):
            raise ValueError(
                f"generate_fn returned {len(outputs)} outputs for "
                f"{len(pairs)} pairs"
            )

        return [
            extract_continuation_logprobs(out.prompt_logprobs, clen, cand)
            for out, clen, cand in zip(outputs, context_lens, candidates)
        ]

    return _score_batch
