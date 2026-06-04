# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling-without-replacement over a fixed allowed token set.

Used by engine-resident choice ranking (vllm/v1/engine/choice_rank.py) to
fuse an all-single-token rank into ONE decode request: at every step the
logits are masked to the allowed set minus the tokens already emitted, so
greedy decoding picks the best remaining candidate -- the k-step selection
loop becomes k ordinary decode steps.

Configured per request via
``SamplingParams.extra_args["shrinking_allowed_token_ids"]`` (list of token
ids). Requests without the key are untouched.

NOTE: not active under speculative decoding -- the coordinator only creates
such requests when ``speculative_config is None`` (draft rows would need
progressive masking that this processor does not implement).
"""

from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

EXTRA_ARGS_KEY = "shrinking_allowed_token_ids"


class ShrinkingAllowedTokenIdsLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        self.device = device
        # index -> (allowed token ids, live ref to the output tokens list)
        self.reqs: dict[int, tuple[tuple[int, ...], list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        """Masking the vocab changes the argmax by design."""
        return False

    @staticmethod
    def _new_state(
        params: SamplingParams,
        _prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> tuple[tuple[int, ...], list[int]] | None:
        extra = params.extra_args
        ids = extra.get(EXTRA_ARGS_KEY) if extra else None
        if not ids:
            return None
        return tuple(ids), output_tok_ids

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self.reqs, batch_update, self._new_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.reqs:
            return logits
        for index, (allowed, out_tok_ids) in self.reqs.items():
            emitted = set(out_tok_ids)
            remaining = [t for t in allowed if t not in emitted]
            if not remaining:
                # max_tokens caps the request at the pool size, so this
                # should not be reached; leave the row untouched.
                continue
            row = logits[index]
            keep = torch.tensor(remaining, dtype=torch.long, device=row.device)
            kept_values = row.index_select(0, keep).clone()
            row.fill_(float("-inf"))
            row.index_copy_(0, keep, kept_values)
        return logits
