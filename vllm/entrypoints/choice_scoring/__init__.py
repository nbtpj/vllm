# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teacher-forced choice scoring and ranking.

This package implements two engine features that share a single primitive --
teacher-forced continuation log-probabilities:

* ``score`` -- given a prompt and a set of choices, return the per-token
  log-probability of every choice (plus sum / mean aggregates, a greedy flag
  and the best choice).
* ``rank`` -- autoregressively select an ordered list of multi-token "bundles"
  from a per-prompt candidate pool, picking the highest average-log-prob
  candidate at each step and shrinking the pool, for ``k`` steps.

The orchestration logic in :mod:`vllm.entrypoints.choice_scoring.core` is pure
Python and depends only on a caller-supplied ``score_fn``; this keeps every
selection / aggregation rule unit-testable without a model or GPU.
"""

from vllm.entrypoints.choice_scoring.params import (
    Candidate,
    ChoiceScore,
    RankOutput,
    RankStep,
    ScoreChoicesOutput,
)

__all__ = [
    "Candidate",
    "ChoiceScore",
    "RankOutput",
    "RankStep",
    "ScoreChoicesOutput",
]
