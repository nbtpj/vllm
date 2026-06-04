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
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
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
SCORE_EXTRA_ARGS_KEY = "choice_score"
# Separator that cannot appear in client-supplied request ids.
_CHILD_SEP = "\x00cr"

# Abuse guards for payloads arriving via public endpoints.
MAX_CANDIDATES = 1024
MAX_CANDIDATE_TOKENS = 4096

# Sentinel candidate index marking a fast (single-child) step: one request
# scores every remaining single-token candidate via logprob_token_ids.
FAST_STEP = -1

# Sentinel marking a decode-fused run: ONE decode request performs every
# remaining selection step (greedy decoding over a shrinking allowed set).
DECODE_RUN = -2

# Sentinel vocab rank for "not the argmax" on the fast step (exact rank is
# unknown; ranks only feed is_greedy-style checks downstream).
NOT_GREEDY_RANK = 2


def get_choice_payload(
    request: Request,
) -> tuple[str, dict[str, Any]] | None:
    """Return ("rank"|"score", payload) if this request carries one."""
    params = request.sampling_params
    if params is None or not params.extra_args:
        return None
    payload = params.extra_args.get(EXTRA_ARGS_KEY)
    if isinstance(payload, dict):
        return "rank", payload
    payload = params.extra_args.get(SCORE_EXTRA_ARGS_KEY)
    if isinstance(payload, dict):
        return "score", payload
    return None


def validate_payload(payload: dict[str, Any], mode: str = "rank") -> str | None:
    """Return an error string for an invalid payload, else None."""
    candidates = payload.get("candidates")
    k = payload.get("k") if mode == "rank" else 1
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
    """In-core state for one rank or score request."""

    def __init__(self, parent: Request, payload: dict[str, Any], mode: str = "rank"):
        self.mode = mode
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
        k: int = payload["k"] if mode == "rank" else 1
        self.select_by: str = payload.get("select_by", "mean")
        self.truncated = k > len(self.remaining)
        self.target = min(k, len(self.remaining))
        # Score groups score multi-token pools in two waves: wave 0 primes
        # the shared-context KV with one candidate; wave 1 siblings hit it.
        self.phase = 0

        self.step = 0
        self.selected: list[dict[str, Any]] = []
        # Decode-fused run accumulators (token, logprob, rank per step).
        self.decode_acc: list[tuple[int, float, int]] = []
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
        kind = get_choice_payload(request)
        if kind is None:
            return False
        mode, payload = kind
        error = validate_payload(payload, mode)
        if error is None and request.prompt_token_ids is None:
            error = "choice scoring requires a token-ids prompt"
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

        group = _RankGroup(request, payload, mode)
        self.groups[group.parent_id] = group
        if mode == "score":
            self._spawn_score(group)
        else:
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
        # Single-token planning. The coordinator decides per step, so mixed
        # pools upgrade dynamically once their multi-token bundles have been
        # selected:
        # - decode-fused run: when every remaining candidate is a unique
        #   single token and >= 2 steps remain, ONE decode request performs
        #   all remaining steps (greedy decoding over a shrinking allowed
        #   set -- the rank loop becomes ordinary decode steps on the
        #   CUDA-graph path). Gated off under speculative decoding (the
        #   masking processor has no draft-row support).
        # - fast step: otherwise, when all remaining are single tokens, one
        #   request with logprob_token_ids scores the pool in one forward.
        ids = [c[0] for _, c in group.remaining]
        all_single = all(len(c) == 1 for _, c in group.remaining) and all(
            t != 0 for t in ids
        )
        steps_left = group.target - group.step
        if (
            all_single
            and steps_left >= 2
            and len(set(ids)) == len(ids)
            and self.core.vllm_config.speculative_config is None
        ):
            self._spawn_decode_run(group, ids, steps_left)
            return
        if all_single and len(set(ids)) <= MAX_LOGPROB_TOKEN_IDS:
            self._spawn_fast_step(group, ids)
            return
        self._spawn_scoring_children(group, group.remaining)

    def _spawn_score(self, group: _RankGroup) -> None:
        """Spawn the (single) scoring wave plan for a score group."""
        ids = [c[0] for _, c in group.remaining]
        if (
            all(len(c) == 1 for _, c in group.remaining)
            and all(t != 0 for t in ids)
            and len(set(ids)) <= MAX_LOGPROB_TOKEN_IDS
        ):
            self._spawn_fast_step(group, ids)
            return
        if group.phase == 0 and len(group.remaining) >= 2 and len(group.context) >= 32:
            # Wave 0: prime the shared-context KV with one candidate; the
            # siblings (wave 1) then hit the cached blocks instead of all
            # re-prefilling the context in parallel.
            self._spawn_scoring_children(group, group.remaining[:1])
            return
        unscored = [it for it in group.remaining if it[0] not in group.results]
        self._spawn_scoring_children(group, unscored)

    def _spawn_scoring_children(
        self, group: _RankGroup, items: list[tuple[int, list[int]]]
    ) -> None:
        context_len = len(group.context)
        for orig_idx, cand in items:
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

    def _spawn_decode_run(
        self, group: _RankGroup, ids: list[int], steps_left: int
    ) -> None:
        child_id = f"{group.parent_id}{_CHILD_SEP}{group.step}.decode"
        child_ecr = EngineCoreRequest(
            request_id=child_id,
            prompt_token_ids=list(group.context),
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=steps_left,
                temperature=0.0,
                logprobs=0,
                ignore_eos=True,
                detokenize=False,
                extra_args={"shrinking_allowed_token_ids": list(ids)},
            ),
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
        group.pending[child_id] = DECODE_RUN
        group.decode_acc.clear()
        self.core.scheduler.add_request(child)

    def _spawn_fast_step(self, group: _RankGroup, ids: list[int]) -> None:
        child_id = f"{group.parent_id}{_CHILD_SEP}{group.step}.fast"
        child_ecr = EngineCoreRequest(
            request_id=child_id,
            prompt_token_ids=list(group.context),
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=1,
                temperature=0.0,
                logprob_token_ids=sorted(set(ids)),
                detokenize=False,
            ),
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
        group.pending[child_id] = FAST_STEP
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

        pending_idx = group.pending.get(out.request_id)
        if pending_idx == DECODE_RUN:
            self._consume_decode_output(group, out)
            return
        if pending_idx == FAST_STEP:
            # Fast step: one child scored every remaining single-token
            # candidate via logprob_token_ids (sample logprobs, row 0).
            lists = out.new_logprobs
            if lists is not None and len(lists.logprob_token_ids):
                lp_map = {
                    int(t): float(lp)
                    for t, lp in zip(lists.logprob_token_ids[0], lists.logprobs[0])
                }
                sampled = out.new_token_ids[0] if out.new_token_ids else None
                ok = True
                for orig_idx, cand in group.remaining:
                    logprob = lp_map.get(cand[0])
                    if logprob is None:
                        ok = False
                        break
                    rank = 1 if cand[0] == sampled else NOT_GREEDY_RANK
                    group.results[orig_idx] = ([logprob], [rank])
                if not ok:
                    group.results.clear()
        else:
            tensors = out.new_prompt_logprobs_tensors
            if tensors is not None and pending_idx is not None:
                # Windowed rows: row j is candidate token j. Column 0 holds
                # the target token's logprob; selected_token_ranks its rank.
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
        scored = (
            len(group.results) == len(group.remaining)
            if orig_idx == FAST_STEP
            else orig_idx in group.results
        )
        if not scored:
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

        if group.mode == "score":
            unscored = [it for it in group.remaining if it[0] not in group.results]
            if unscored:
                group.phase = 1
                self._spawn_scoring_children(group, unscored)
            else:
                self._finalize(group)
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

    def _consume_decode_output(self, group: _RankGroup, out: EngineCoreOutput) -> None:
        """Accumulate one decode-fused child's streamed steps; finalize at
        the end of the run."""
        lists = out.new_logprobs
        n_new = len(out.new_token_ids)
        if n_new and lists is not None and len(lists.logprob_token_ids) == n_new:
            for j, token in enumerate(out.new_token_ids):
                group.decode_acc.append(
                    (
                        int(token),
                        float(lists.logprobs[j][0]),
                        int(lists.sampled_token_ranks[j]),
                    )
                )
        if not out.finished:
            return

        group.pending.pop(out.request_id, None)
        self.child_to_parent.pop(out.request_id, None)
        token_to_remaining = {c[0]: i for i, c in group.remaining}
        expected = group.target - group.step
        if len(group.decode_acc) != expected:
            self._finalize(
                group,
                error=f"decode-fused rank produced {len(group.decode_acc)} "
                f"steps, expected {expected}",
            )
            return
        for token, logprob, rank in group.decode_acc:
            orig_idx = token_to_remaining.pop(token, None)
            if orig_idx is None:
                self._finalize(
                    group,
                    error=f"decode-fused rank emitted unexpected token {token}",
                )
                return
            group.selected.append(
                {
                    "order": group.step,
                    "choice_index": orig_idx,
                    "token_ids": [token],
                    "token_logprobs": [logprob],
                    "ranks": [rank],
                }
            )
            group.step += 1
        self._finalize(group)

    def _finalize(self, group: _RankGroup, error: str | None = None) -> None:
        if error is not None:
            result: dict[str, Any] = {"error": error}
        elif group.mode == "score":
            result = {
                "choices": [
                    {
                        "index": orig_idx,
                        "token_ids": list(cand),
                        "token_logprobs": group.results[orig_idx][0],
                        "ranks": group.results[orig_idx][1],
                    }
                    for orig_idx, cand in group.remaining
                ]
            }
        else:
            result = {
                "selected": group.selected,
                "truncated": group.truncated,
            }
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
