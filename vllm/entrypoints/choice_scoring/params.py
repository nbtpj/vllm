# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure data structures for choice scoring and ranking.

These intentionally depend on nothing from the rest of vLLM (no torch, no
engine types) so that the orchestration logic in :mod:`core` can be exercised
in fast, deterministic unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# How a choice's scalar selection score is derived from its per-token logprobs.
SelectBy = Literal["mean", "sum"]


@dataclass
class Candidate:
    """A single choice / candidate "bundle" of one or more tokens.

    Exactly one of ``token_ids`` (already tokenized) is required for scoring;
    ``text`` and ``tokens`` are optional metadata carried through to the output
    for convenience and human readability.
    """

    token_ids: list[int]
    text: str | None = None
    tokens: list[str] | None = None
    # Stable index into the *original* candidate list for this prompt. Used so
    # that rank output can refer back to the input choice regardless of how the
    # pool shrinks.
    index: int = 0


@dataclass
class ChoiceScore:
    """Teacher-forced score of one choice given a fixed context."""

    index: int
    token_ids: list[int]
    token_logprobs: list[float]
    sum_logprob: float
    mean_logprob: float
    # ``True`` if the choice was the argmax continuation at *every* position
    # (i.e. every token had vocab rank 1). ``None`` when ranks were not
    # requested from the engine.
    is_greedy: bool | None = None
    tokens: list[str] | None = None
    text: str | None = None


@dataclass
class ScoreChoicesOutput:
    """Result of scoring all choices for a single prompt."""

    prompt_token_ids: list[int]
    choices: list[ChoiceScore]
    # Index (into ``choices`` / the original choice list) of the best choice by
    # the configured ``select_by`` metric, with lowest-index tie-breaking.
    # ``-1`` only if ``choices`` is empty.
    best_choice_index: int


@dataclass
class RankStep:
    """One selected bundle in a rank result."""

    # 0-based position in the output ordering (0 == picked first / best).
    order: int
    # Index into the original candidate list for this prompt.
    choice_index: int
    token_ids: list[int]
    token_logprobs: list[float]
    sum_logprob: float
    mean_logprob: float
    tokens: list[str] | None = None
    text: str | None = None


@dataclass
class RankOutput:
    """Ordered selection result for a single prompt."""

    prompt_token_ids: list[int]
    selected: list[RankStep] = field(default_factory=list)
    # ``True`` when ``k`` exceeded the candidate pool size and selection stopped
    # early because the pool was exhausted.
    truncated: bool = False
