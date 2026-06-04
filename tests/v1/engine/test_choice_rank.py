# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-resident choice-rank coordinator (no model / GPU).

A fake EngineCore (scheduler stub) lets us drive the coordinator end to end:
parent interception, child fabrication, out-of-order child completion,
selection parity against the client-side reference driver, truncation,
payload validation, and abort teardown.
"""

from __future__ import annotations

import torch

from vllm.entrypoints.choice_scoring.batching import rank_batch
from vllm.entrypoints.choice_scoring.params import Candidate
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.choice_rank import (
    ChoiceRankCoordinator,
    get_choice_rank_payload,
    validate_payload,
)
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.request import Request


class FakeScheduler:
    def __init__(self):
        self.queue: list[Request] = []
        self.finished: list[str] = []

    def add_request(self, request: Request):
        self.queue.append(request)

    def finish_requests(self, request_ids, status):
        self.finished.extend(request_ids)
        self.queue = [r for r in self.queue if r.request_id not in request_ids]


class FakeCore:
    def __init__(self):
        self.scheduler = FakeScheduler()
        self.request_block_hasher = None


# Per-token logprob assigned to each token id by the fake model.
TABLE = {10: -3.0, 11: -1.0, 12: -2.0, 20: -0.5, 21: -4.0, 22: -1.5}


def make_parent(prompt, candidates, k, select_by="mean", request_id="parent-0"):
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        detokenize=False,
        extra_args={
            "choice_rank": {
                "candidates": candidates,
                "k": k,
                "select_by": select_by,
            }
        },
    )
    ecr = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return Request.from_engine_core_request(ecr, None)


def child_output(child: Request, finished=True) -> EngineCoreOutput:
    """Fabricate the output the runner would produce for a child."""
    import numpy as np

    from vllm.v1.engine import FinishReason
    from vllm.v1.outputs import LogprobsLists

    sp = child.sampling_params
    assert sp is not None
    if sp.logprob_token_ids:
        # Fast step: sample logprobs for the requested ids (+ sampled).
        ids = list(sp.logprob_token_ids)
        sampled = max(ids, key=lambda t: TABLE[t])
        row_ids = [sampled, *ids]
        row_lps = [TABLE[t] for t in row_ids]
        lists = LogprobsLists(
            logprob_token_ids=np.array([row_ids]),
            logprobs=np.array([row_lps]),
            sampled_token_ranks=np.array([1]),
        )
        return EngineCoreOutput(
            request_id=child.request_id,
            new_token_ids=[sampled],
            new_logprobs=lists,
            finish_reason=FinishReason.LENGTH if finished else None,
        )
    assert sp.prompt_logprobs_from is not None
    cand = child.prompt_token_ids[sp.prompt_logprobs_from :]
    logprobs = torch.tensor([[TABLE[t]] for t in cand], dtype=torch.float32)
    tensors = LogprobsTensors(
        logprob_token_ids=torch.tensor([[t] for t in cand]),
        logprobs=logprobs,
        selected_token_ranks=torch.tensor([2] * len(cand)),
    )
    from vllm.v1.engine import FinishReason

    return EngineCoreOutput(
        request_id=child.request_id,
        new_token_ids=[0],
        new_prompt_logprobs_tensors=tensors,
        finish_reason=FinishReason.LENGTH if finished else None,
    )


def drive(core, coordinator, batch=None):
    """Complete queued children one batch at a time; return parent outputs."""
    parent_outputs = []
    while core.scheduler.queue:
        children = list(core.scheduler.queue)
        if batch is not None:
            children = children[:batch]
        outs = [child_output(c) for c in children]
        core.scheduler.queue = [r for r in core.scheduler.queue if r not in children]
        result = coordinator.process_outputs({0: EngineCoreOutputs(outputs=outs)})
        for ecos in result.values():
            parent_outputs.extend(o for o in ecos.outputs if o.choice_rank_result)
    return parent_outputs


def reference_orders(prompt, candidates, k, select_by="mean"):
    def score_fn(pairs):
        return [([TABLE[t] for t in cand], None) for _ctx, cand in pairs]

    cands = [Candidate(token_ids=c, index=i) for i, c in enumerate(candidates)]
    out = rank_batch([list(prompt)], [cands], [k], score_fn, select_by)[0]
    return [s.choice_index for s in out.selected], out.truncated


def test_intercept_and_full_rank_matches_reference():
    core = FakeCore()
    coord = ChoiceRankCoordinator(core)
    candidates = [[10, 11], [20], [12, 22, 11]]
    parent = make_parent([1, 2, 3], candidates, k=3)
    assert coord.try_intercept(parent)
    # Step 0 children are queued, parent is not.
    assert len(core.scheduler.queue) == 3
    assert all("\x00cr" in r.request_id for r in core.scheduler.queue)
    # Children share the context and request the window at its length.
    child = core.scheduler.queue[0]
    assert child.prompt_token_ids[:3] == [1, 2, 3]
    assert child.sampling_params.prompt_logprobs_from == 3

    outs = drive(core, coord)
    assert len(outs) == 1
    result = outs[0].choice_rank_result
    orders = [s["choice_index"] for s in result["selected"]]
    ref_orders, ref_trunc = reference_orders([1, 2, 3], candidates, 3)
    assert orders == ref_orders
    assert result["truncated"] is ref_trunc
    # Selected steps carry per-token logprobs of the chosen bundle.
    for step in result["selected"]:
        cand = candidates[step["choice_index"]]
        assert step["token_ids"] == cand
        assert step["token_logprobs"] == [TABLE[t] for t in cand]
    # All bookkeeping cleared.
    assert not coord.groups and not coord.child_to_parent


def test_select_by_sum_matches_reference():
    candidates = [[10, 11], [20], [12, 22, 11]]
    for select_by in ("mean", "sum"):
        core = FakeCore()
        coord = ChoiceRankCoordinator(core)
        coord.try_intercept(make_parent([1], candidates, k=2, select_by=select_by))
        outs = drive(core, coord, batch=1)  # one child per poll: out of order
        orders = [s["choice_index"] for s in outs[0].choice_rank_result["selected"]]
        ref_orders, _ = reference_orders([1], candidates, 2, select_by)
        assert orders == ref_orders, select_by


def test_truncation_when_k_exceeds_pool():
    core = FakeCore()
    coord = ChoiceRankCoordinator(core)
    coord.try_intercept(make_parent([1], [[10], [11]], k=5))
    outs = drive(core, coord)
    result = outs[0].choice_rank_result
    assert len(result["selected"]) == 2
    assert result["truncated"] is True


def test_invalid_payload_yields_error_result():
    core = FakeCore()
    coord = ChoiceRankCoordinator(core)
    assert coord.try_intercept(make_parent([1], [], k=1))
    assert not core.scheduler.queue
    assert coord.has_ready_outputs
    result = coord.process_outputs({})
    out = result[0].outputs[0]
    assert "error" in out.choice_rank_result
    assert not coord.has_ready_outputs


def test_non_rank_request_passes_through():
    core = FakeCore()
    coord = ChoiceRankCoordinator(core)
    ecr = EngineCoreRequest(
        request_id="normal",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    request = Request.from_engine_core_request(ecr, None)
    assert get_choice_rank_payload(request) is None
    assert not coord.try_intercept(request)


def test_child_outputs_filtered_from_client_stream():
    core = FakeCore()
    coord = ChoiceRankCoordinator(core)
    coord.try_intercept(make_parent([1], [[10], [11]], k=1))
    children = list(core.scheduler.queue)
    core.scheduler.queue = []
    outs = [child_output(c) for c in children]
    # Add an unrelated output that must pass through untouched.
    passthrough = EngineCoreOutput(request_id="other", new_token_ids=[7])
    ecos = EngineCoreOutputs(outputs=[passthrough, *outs])
    ecos.finished_requests = {"other", children[0].request_id}
    result = coord.process_outputs({0: ecos})
    ids = [o.request_id for o in result[0].outputs]
    assert "other" in ids
    assert all("\x00cr" not in i for i in ids)
    assert result[0].finished_requests == {"other"}
    # Parent result was attached in the same batch.
    assert any(o.choice_rank_result for o in result[0].outputs)


def test_abort_parent_tears_down_children():
    core = FakeCore()
    coord = ChoiceRankCoordinator(core)
    coord.try_intercept(make_parent([1], [[10, 11], [11, 12], [12, 20]], k=2))
    assert len(core.scheduler.queue) == 3
    coord.handle_aborts(["parent-0"])
    assert not coord.groups
    assert len(core.scheduler.finished) == 3
    assert not core.scheduler.queue


def test_child_failure_produces_error_result():
    from vllm.v1.engine import FinishReason

    core = FakeCore()
    coord = ChoiceRankCoordinator(core)
    coord.try_intercept(make_parent([1], [[10, 11], [11, 12]], k=1))
    children = list(core.scheduler.queue)
    assert len(children) == 2
    core.scheduler.queue = []
    # First child finishes WITHOUT logprobs (e.g. internal abort).
    bad = EngineCoreOutput(
        request_id=children[0].request_id,
        new_token_ids=[],
        finish_reason=FinishReason.ABORT,
    )
    result = coord.process_outputs({0: EngineCoreOutputs(outputs=[bad])})
    parent_outs = [o for o in result[0].outputs if o.choice_rank_result]
    assert parent_outs and "error" in parent_outs[0].choice_rank_result
    # Remaining child was aborted with the group.
    assert children[1].request_id in core.scheduler.finished


def test_validate_payload_messages():
    assert validate_payload({"candidates": [[1]], "k": 1}) is None
    assert "non-empty" in validate_payload({"candidates": [], "k": 1})
    assert "token ids" in validate_payload({"candidates": [[]], "k": 1})
    assert "k must" in validate_payload({"candidates": [[1]], "k": 0})
    assert "select_by" in validate_payload(
        {"candidates": [[1]], "k": 1, "select_by": "max"}
    )


def test_single_token_pool_uses_fast_step():
    core = FakeCore()
    coord = ChoiceRankCoordinator(core)
    candidates = [[10], [11], [12]]
    coord.try_intercept(make_parent([1, 2], candidates, k=3))
    # One fast child per step instead of pool-size children.
    assert len(core.scheduler.queue) == 1
    child = core.scheduler.queue[0]
    assert child.sampling_params.logprob_token_ids == [10, 11, 12]
    assert child.prompt_token_ids == [1, 2]

    outs = drive(core, coord)
    result = outs[0].choice_rank_result
    orders = [s["choice_index"] for s in result["selected"]]
    ref_orders, _ = reference_orders([1, 2], candidates, 3)
    assert orders == ref_orders
    # Per-step logprobs match the table.
    for step in result["selected"]:
        assert step["token_logprobs"] == [TABLE[candidates[step["choice_index"]][0]]]


def test_mixed_pool_upgrades_to_fast_step():
    core = FakeCore()
    coord = ChoiceRankCoordinator(core)
    # One multi-token bundle that wins step 0; the rest single tokens.
    candidates = [[20, 11], [10], [12]]
    coord.try_intercept(make_parent([1], candidates, k=3))
    # Step 0 is teacher-forced (mixed pool): 3 children.
    assert len(core.scheduler.queue) == 3
    children = list(core.scheduler.queue)
    core.scheduler.queue = []
    coord.process_outputs(
        {0: EngineCoreOutputs(outputs=[child_output(c) for c in children])}
    )
    # [20, 11] (mean -0.75) won; remaining pool is all single-token ->
    # step 1 upgrades to a single fast child.
    assert len(core.scheduler.queue) == 1
    assert core.scheduler.queue[0].sampling_params.logprob_token_ids == [10, 12]
    outs = drive(core, coord)
    orders = [s["choice_index"] for s in outs[0].choice_rank_result["selected"]]
    ref_orders, _ = reference_orders([1], candidates, 3)
    assert orders == ref_orders
