# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client glue for engine-resident rank (see vllm/v1/engine/choice_rank.py).

The client sends ONE request per prompt: the prompt is the shared context and
``SamplingParams.extra_args["choice_rank"]`` carries the candidate pool and
``k``. The engine runs the whole selection loop internally and returns the
structured result on ``RequestOutput.choice_rank_result``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.choice_scoring.params import (
    Candidate,
    RankOutput,
    RankStep,
)
from vllm.sampling_params import SamplingParams


def make_native_rank_params(
    candidates: Sequence[Candidate],
    k: int,
    select_by: str,
) -> SamplingParams:
    """SamplingParams for one engine-resident rank parent request."""
    payload: dict[str, Any] = {
        "candidates": [list(c.token_ids) for c in candidates],
        "k": int(k),
        "select_by": select_by,
    }
    return SamplingParams(
        max_tokens=1,
        temperature=0.0,
        detokenize=False,
        extra_args={"choice_rank": payload},
    )


def build_rank_output(
    prompt_token_ids: Sequence[int],
    candidates: Sequence[Candidate],
    result: dict[str, Any] | None,
) -> RankOutput:
    """Convert an engine ``choice_rank_result`` dict into a RankOutput.

    Raises:
        ValueError: if the engine reported an error for this request.
        RuntimeError: if the engine returned no result (engine-resident
            rank unsupported by the connected engine).
    """
    if result is None:
        raise RuntimeError(
            "engine returned no choice_rank_result; the connected engine "
            "does not support engine-resident rank (unset "
            "VLLM_ENABLE_NATIVE_CHOICE_SCORING to use the client-side path)"
        )
    if "error" in result:
        raise ValueError(f"engine rejected choice_rank request: {result['error']}")

    selected = []
    for step in result["selected"]:
        choice_index = step["choice_index"]
        cand = candidates[choice_index]
        logprobs = [float(lp) for lp in step["token_logprobs"]]
        selected.append(
            RankStep(
                order=int(step["order"]),
                choice_index=choice_index,
                token_ids=list(step["token_ids"]),
                token_logprobs=logprobs,
                sum_logprob=sum(logprobs),
                mean_logprob=sum(logprobs) / len(logprobs),
                tokens=cand.tokens,
                text=cand.text,
            )
        )
    return RankOutput(
        prompt_token_ids=list(prompt_token_ids),
        selected=selected,
        truncated=bool(result["truncated"]),
    )
