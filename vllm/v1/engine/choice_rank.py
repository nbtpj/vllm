# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-resident choice ranking (native L3, general bundles).

A *rank* request arrives at the EngineCore as one ordinary request whose
``SamplingParams.extra_args["choice_rank"]`` carries the whole candidate pool
(arbitrary multi-token bundles) and ``k``. The coordinator intercepts it
before scheduling and drives the entire autoregressive selection loop inside
the engine:

1. For each remaining candidate, fabricate a child scoring request
   ``context + candidate`` with a windowed prompt-logprobs request
   (``prompt_logprobs_from = len(context)``), sharing the context KV via the
   prefix cache.
2. Consume the children's windowed logprob tensors directly in-core (no IPC
   serialization, no client-side pythonization), select the best candidate
   (sum/mean, lowest-original-index tie-break -- identical semantics to
   ``vllm.entrypoints.choice_scoring``), extend the context, and immediately
   schedule the next step.
3. After ``k`` selections (or pool exhaustion), emit a single
   ``EngineCoreOutput`` for the parent carrying the structured result in
   ``choice_rank_result``.

Child requests are normal scheduler requests: they preempt/resume like any
other and write/read the prefix cache, so this works for any model and any
bundle lengths -- no custom attention masks required.

Payload schema (``extra_args["choice_rank"]``)::

    {"candidates": [[tok, ...], ...], "k": int, "select_by": "mean" | "sum"}

Result schema (``EngineCoreOutput.choice_rank_result``)::

    {"selected": [{"order": int, "choice_index": int, "token_ids": [int],
                   "token_logprobs": [float], "ranks": [int]}, ...],
     "truncated": bool}
    or {"error": str} if the payload was invalid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCore

logger = init_logger(__name__)

EXTRA_ARGS_KEY = "choice_rank"
# Separator that cannot appear in client-supplied request ids.
_CHILD_SEP = "\x00cr"

# Abuse guards for payloads arriving via public endpoints.
MAX_CANDIDATES = 1024
MAX_CANDIDATE_TOKENS = 4096


def get_choice_rank_payload(request: Request) -> dict[str, Any] | None:
    """Return the choice-rank payload if this request carries one."""
    params = request.sampling_params
    if params is None or not params.extra_args:
        return None
    payload = params.extra_args.get(EXTRA_ARGS_KEY)
    return payload if isinstance(payload, dict) else None


def validate_payload(payload: dict[str, Any]) -> str | None:
    """Return an error string for an invalid payload, else None."""
    candidates = payload.get("candidates")
    k = payload.get("k")
    select_by = payload.get("select_by", "mean")
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) > MAX_CANDIDATES
    ):
        return f"candidates must be a non-empty list of <= {MAX_CANDIDATES}"
    for cand in candidates:
        if (
            not isinstance(cand, list)
            or not cand
            or len(cand) > MAX_CANDIDATE_TOKENS
            or not all(isinstance(t, int) and t >= 0 for t in cand)
        ):
            return "each candidate must be a non-empty list of token ids"
    if not isinstance(k, int) or k < 1:
        return "k must be an int >= 1"
    if select_by not in ("mean", "sum"):
        return "select_by must be 'mean' or 'sum'"
    return None


class _RankGroup:
    """In-core state for one rank request."""

    def __init__(self, parent: Request, payload: dict[str, Any]):
        self.parent_id = parent.request_id
        self.client_index = parent.client_index
        self.arrival_time = parent.arrival_time
        self.lora_request = parent.lora_request
        self.priority = parent.priority

        assert parent.prompt_token_ids is not None
        self.context: list[int] = list(parent.prompt_token_ids)
        candidates: list[list[int]] = payload["candidates"]
        self.remaining: list[tuple[int, list[int]]] = [
            (i, list(c)) for i, c in enumerate(candidates)
        ]
        k: int = payload["k"]
        self.select_by: str = payload.get("select_by", "mean")
        self.truncated = k > len(self.remaining)
        self.target = min(k, len(self.remaining))

        self.step = 0
        self.selected: list[dict[str, Any]] = []
        # child_id -> original candidate index (this step)
        self.pending: dict[str, int] = {}
        # original candidate index -> (token_logprobs, ranks)
        self.results: dict[int, tuple[list[float], list[int]]] = {}

    def select_best(self) -> int:
        """Pick the best scored candidate (lowest-index tie-break).

        Mirrors ``entrypoints.choice_scoring.core.select_best_index``:
        iterate candidates in original-index order, strict ``>`` keeps the
        first (lowest-index) maximum.
        """
        best_idx = -1
        best_score = float("-inf")
        for orig_idx, _ in self.remaining:
            logprobs, _ranks = self.results[orig_idx]
            score = sum(logprobs)
            if self.select_by == "mean":
                score /= len(logprobs)
            if score > best_score:
                best_score = score
                best_idx = orig_idx
        return best_idx


class ChoiceRankCoordinator:
    """Drives engine-resident rank groups inside ``EngineCore``."""

    def __init__(self, core: EngineCore):
        self.core = core
        self.groups: dict[str, _RankGroup] = {}
        self.child_to_parent: dict[str, str] = {}
        # Parent outputs ready to be attached to the next outputs batch.
        self._ready: list[tuple[int, EngineCoreOutput]] = []
        # Ids of children aborted engine-side whose finished_requests
        # entries may still surface in later output batches.
        self._zombie_children: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self.groups) or bool(self._ready)

    @property
    def has_ready_outputs(self) -> bool:
        """Parent results waiting to be flushed (needs a step() call even
        when the scheduler is empty)."""
        return bool(self._ready)

    # ------------------------------------------------------------------ #
    # request interception
    # ------------------------------------------------------------------ #
    def try_intercept(self, request: Request) -> bool:
        """If ``request`` is a rank parent, take ownership and return True."""
        payload = get_choice_rank_payload(request)
        if payload is None:
            return False
        error = validate_payload(payload)
        if error is None and request.prompt_token_ids is None:
            error = "choice_rank requires a token-ids prompt"
        if error is not None:
            logger.warning(
                "Rejecting choice_rank request %s: %s", request.request_id, error
            )
            self._ready.append(
                (
                    request.client_index,
                    self._parent_output(request.request_id, {"error": error}),
                )
            )
            return True

        group = _RankGroup(request, payload)
        self.groups[group.parent_id] = group
        self._spawn_step(group)
        return True

    def _child_params(self, context_len: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=0,
            prompt_logprobs_from=context_len,
            detokenize=False,
        )

    def _spawn_step(self, group: _RankGroup) -> None:
        group.pending.clear()
        group.results.clear()
        context_len = len(group.context)
        for orig_idx, cand in group.remaining:
            child_id = f"{group.parent_id}{_CHILD_SEP}{group.step}.{orig_idx}"
            child_ecr = EngineCoreRequest(
                request_id=child_id,
                prompt_token_ids=group.context + cand,
                mm_features=None,
                sampling_params=self._child_params(context_len),
                pooling_params=None,
                arrival_time=group.arrival_time,
                lora_request=group.lora_request,
                cache_salt=None,
                data_parallel_rank=None,
                client_index=group.client_index,
                priority=group.priority,
            )
            child = Request.from_engine_core_request(
                child_ecr, self.core.request_block_hasher
            )
            self.child_to_parent[child_id] = group.parent_id
            group.pending[child_id] = orig_idx
            self.core.scheduler.add_request(child)

    # ------------------------------------------------------------------ #
    # output interception
    # ------------------------------------------------------------------ #
    def process_outputs(
        self, outputs: dict[int, EngineCoreOutputs]
    ) -> dict[int, EngineCoreOutputs]:
        """Consume child outputs, advance groups, attach parent results."""
        if not self:
            return outputs

        # Child ids consumed during this call (they get popped from
        # child_to_parent as their step resolves, but must still be
        # filtered out of this batch's finished_requests sets).
        consumed: set[str] = set()
        for ecos in outputs.values():
            kept: list[EngineCoreOutput] = []
            for out in ecos.outputs:
                parent_id = self.child_to_parent.get(out.request_id)
                if parent_id is None:
                    kept.append(out)
                    continue
                consumed.add(out.request_id)
                self._consume_child_output(parent_id, out)
            if len(kept) != len(ecos.outputs):
                ecos.outputs = kept
            if ecos.finished_requests:
                filtered = set()
                for rid in ecos.finished_requests:
                    if rid in self.child_to_parent or rid in consumed:
                        continue
                    if rid in self._zombie_children:
                        self._zombie_children.discard(rid)
                        continue
                    filtered.add(rid)
                ecos.finished_requests = filtered

        # Attach any parent results that became ready.
        if self._ready:
            ready, self._ready = self._ready, []
            for client_index, parent_out in ready:
                ecos = outputs.get(client_index)
                if ecos is None:
                    ecos = EngineCoreOutputs(outputs=[])
                    outputs[client_index] = ecos
                ecos.outputs.append(parent_out)
        return outputs

    def _consume_child_output(self, parent_id: str, out: EngineCoreOutput) -> None:
        group = self.groups.get(parent_id)
        if group is None:
            # Parent aborted; drop stragglers.
            self.child_to_parent.pop(out.request_id, None)
            return

        tensors = out.new_prompt_logprobs_tensors
        pending_idx = group.pending.get(out.request_id)
        if tensors is not None and pending_idx is not None:
            # Windowed rows: row j is candidate token j. Column 0 holds the
            # target token's logprob; selected_token_ranks its vocab rank.
            group.results[pending_idx] = (
                tensors.logprobs[:, 0].tolist(),
                tensors.selected_token_ranks.tolist(),
            )
        if not out.finished:
            return

        orig_idx = group.pending.pop(out.request_id, None)
        self.child_to_parent.pop(out.request_id, None)
        if orig_idx is None:
            return
        if orig_idx not in group.results:
            # Child finished without producing logprobs (e.g. aborted by
            # the engine). Fail the whole group rather than mis-rank.
            self._finalize(
                group,
                error=f"scoring child failed for candidate "
                f"{orig_idx} at step {group.step}",
            )
            return
        if group.pending:
            return

        # Step complete: select, advance, and either respawn or finalize.
        best = group.select_best()
        logprobs, ranks = group.results[best]
        cand_tokens = next(c for i, c in group.remaining if i == best)
        group.selected.append(
            {
                "order": group.step,
                "choice_index": best,
                "token_ids": list(cand_tokens),
                "token_logprobs": logprobs,
                "ranks": ranks,
            }
        )
        group.context.extend(cand_tokens)
        group.remaining = [(i, c) for i, c in group.remaining if i != best]
        group.step += 1
        if group.step < group.target and group.remaining:
            self._spawn_step(group)
        else:
            self._finalize(group)

    def _finalize(self, group: _RankGroup, error: str | None = None) -> None:
        if error is None:
            result: dict[str, Any] = {
                "selected": group.selected,
                "truncated": group.truncated,
            }
        else:
            result = {"error": error}
        self._abort_group_children(group)
        self.groups.pop(group.parent_id, None)
        self._ready.append(
            (group.client_index, self._parent_output(group.parent_id, result))
        )

    @staticmethod
    def _parent_output(parent_id: str, result: dict[str, Any]) -> EngineCoreOutput:
        return EngineCoreOutput(
            request_id=parent_id,
            new_token_ids=[],
            finish_reason=FinishReason.STOP,
            choice_rank_result=result,
        )

    # ------------------------------------------------------------------ #
    # aborts
    # ------------------------------------------------------------------ #
    def handle_aborts(self, request_ids: list[str]) -> None:
        """Tear down groups whose parents are being aborted."""
        if not self.groups:
            return
        for request_id in request_ids:
            group = self.groups.pop(request_id, None)
            if group is not None:
                self._abort_group_children(group)

    def _abort_group_children(self, group: _RankGroup) -> None:
        if not group.pending:
            return
        child_ids = list(group.pending)
        for child_id in child_ids:
            self.child_to_parent.pop(child_id, None)
            self._zombie_children.add(child_id)
        group.pending.clear()
        self.core.scheduler.finish_requests(child_ids, RequestStatus.FINISHED_ABORTED)
