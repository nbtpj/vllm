# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs


class PromptLogprobsWorker:
    def __init__(self, max_num_reqs: int):
        self.max_num_reqs = max_num_reqs

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        # First prompt token position to produce logprobs for (window start);
        # 0 means "all positions" (SamplingParams.prompt_logprobs_from unset).
        self.prompt_logprobs_from = np.zeros(self.max_num_reqs, dtype=np.int64)
        # req_idx -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}

    def add_request(self, req_id: str, req_idx: int, sampling_params: SamplingParams):
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        self.prompt_logprobs_from[req_idx] = sampling_params.prompt_logprobs_from or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs, max_model_len]
        all_token_ids: torch.Tensor,
        # [max_num_reqs]
        num_computed_tokens: torch.Tensor,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # Common case: No request asks for prompt logprobs.
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_lens = prompt_lens[idx_mapping_np]
        computed_prefill = input_batch.num_computed_prefill_tokens_np
        includes_prompt = computed_prefill < prompt_lens
        # NOTE(woosuk): If the request was resumed after preemption, its prompt
        # logprobs must have been computed before preemption. Skip.
        resumed_after_prompt = prompt_lens < input_batch.prefill_len_np
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # get the maximum number in this batch
        requested_num_prompt_logprobs = num_prompt_logprobs[needs_prompt_logprobs]
        max_num_prompt_logprobs = (
            -1
            if np.any(requested_num_prompt_logprobs == -1)
            else int(requested_num_prompt_logprobs.max())
        )

        # Get the prompt logprobs token_ids.
        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )

        # Optional per-request window (SamplingParams.prompt_logprobs_from):
        # only rows predicting prompt token positions >= the window start are
        # run through the LM head. The row at absolute prompt position p
        # predicts token p+1, so the first wanted row is window_start - 1.
        prompt_logprobs_from = self.prompt_logprobs_from[idx_mapping_np]
        window_active = bool(np.any((prompt_logprobs_from > 0) & needs_prompt_logprobs))

        query_start_loc_np = input_batch.query_start_loc_np
        num_batch_reqs = len(input_batch.req_ids)
        # Per-request [start, end) row range into the tensor handed to the
        # LM head (the full token array on the fast path, the gathered rows
        # on the windowed path).
        req_row_start = np.zeros(num_batch_reqs, dtype=np.int64)
        req_row_end = np.zeros(num_batch_reqs, dtype=np.int64)
        total_rows = -1
        if not window_active:
            # Fast path (original behavior): all scheduled tokens.
            req_row_start[:] = query_start_loc_np[:num_batch_reqs]
            req_row_end[:] = query_start_loc_np[1 : num_batch_reqs + 1]
        else:
            wanted: list[np.ndarray] = []
            total_rows = 0
            for i in range(num_batch_reqs):
                if not needs_prompt_logprobs[i]:
                    continue
                q_start = int(query_start_loc_np[i])
                q_end = int(query_start_loc_np[i + 1])
                skip = 0
                if prompt_logprobs_from[i] > 0:
                    first_row = int(prompt_logprobs_from[i]) - 1
                    skip = min(
                        max(first_row - int(computed_prefill[i]), 0),
                        q_end - q_start,
                    )
                req_row_start[i] = total_rows
                req_row_end[i] = total_rows + (q_end - q_start - skip)
                total_rows = int(req_row_end[i])
                if q_start + skip < q_end:
                    wanted.append(np.arange(q_start + skip, q_end, dtype=np.int64))
            wanted_np = (
                np.concatenate(wanted) if wanted else np.empty(0, dtype=np.int64)
            )
            wanted_idx = torch.from_numpy(wanted_np).to(
                prompt_logprobs_token_ids.device, non_blocking=True
            )
            prompt_logprobs_token_ids = prompt_logprobs_token_ids[wanted_idx]
            hidden_states = hidden_states[: input_batch.num_tokens][wanted_idx]

        if window_active and total_rows == 0:
            # Every needy chunk is entirely below its window this step.
            prompt_token_ids = prompt_logprobs = prompt_ranks = None
        else:
            prompt_token_ids, prompt_logprobs, prompt_ranks = (
                compute_prompt_logprobs_with_chunking(
                    prompt_logprobs_token_ids,
                    hidden_states
                    if window_active
                    else hidden_states[: input_batch.num_tokens],
                    logits_fn,
                    max_num_prompt_logprobs,
                )
            )

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens

        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            req_is_prompt_chunked = is_prompt_chunked[i]
            req_num_prompt_logprobs = int(num_prompt_logprobs[i])
            start_idx = int(req_row_start[i])
            end_idx = int(req_row_end[i])
            if not req_is_prompt_chunked:
                end_idx -= 1

            # no logprobs if start_idx >= end_idx (or nothing was computed
            # because every needy chunk was below its window this step)
            if prompt_logprobs is None or start_idx >= end_idx:
                logprobs = None
            else:
                width = (
                    prompt_logprobs.shape[1]
                    if req_num_prompt_logprobs == -1
                    else req_num_prompt_logprobs + 1
                )
                logprobs = LogprobsTensors(
                    logprob_token_ids=prompt_token_ids[start_idx:end_idx, :width],
                    logprobs=prompt_logprobs[start_idx:end_idx, :width],
                    selected_token_ranks=prompt_ranks[start_idx:end_idx],
                )

            prompt_logprobs_list = self.in_progress_prompt_logprobs[req_id]
            if logprobs is not None and (req_is_prompt_chunked or prompt_logprobs_list):
                prompt_logprobs_list.append(logprobs)
            if req_is_prompt_chunked:
                # Prompt is chunked. Do not return the logprobs yet.
                continue

            if prompt_logprobs_list:
                # Merge the in-progress logprobs.
                logprobs = LogprobsTensors(
                    logprob_token_ids=torch.cat(
                        [x.logprob_token_ids for x in prompt_logprobs_list]
                    ),
                    logprobs=torch.cat([x.logprobs for x in prompt_logprobs_list]),
                    selected_token_ranks=torch.cat(
                        [x.selected_token_ranks for x in prompt_logprobs_list]
                    ),
                )
                prompt_logprobs_list.clear()

            if logprobs is None:
                continue

            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        # NOTE(woosuk): We should shift the pos by one
        # because the logprob is computed for the next token.
        target_pos = num_computed_tokens + 1 + block
        token_ids = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + target_pos,
            mask=mask,
        )
        tl.store(
            prompt_logprobs_token_ids_ptr + query_start + block, token_ids, mask=mask
        )


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=idx_mapping.device)
    num_reqs = idx_mapping.shape[0]
    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    return token_ids


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks.
    CHUNK_SIZE = 1024
    token_ids = []
    logprobs = []
    ranks = []
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], CHUNK_SIZE):
        end_idx = start_idx + CHUNK_SIZE
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        requested_num_prompt_logprobs = (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        prompt_logprobs = compute_topk_logprobs(
            prompt_logits,
            requested_num_prompt_logprobs,
            prompt_token_ids[start_idx:end_idx],
        )
        token_ids.append(prompt_logprobs.logprob_token_ids)
        logprobs.append(prompt_logprobs.logprobs)
        ranks.append(prompt_logprobs.selected_token_ranks)

    token_ids = torch.cat(token_ids, dim=0) if len(token_ids) > 1 else token_ids[0]
    logprobs = torch.cat(logprobs, dim=0) if len(logprobs) > 1 else logprobs[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return token_ids, logprobs, ranks
