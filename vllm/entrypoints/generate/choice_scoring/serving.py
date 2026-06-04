# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP serving for choice scoring (``/score_choices``) and ranking (``/rank``).

Both endpoints work on any generative model. They build on the unit-tested,
model-independent orchestration in :mod:`vllm.entrypoints.choice_scoring` and
the existing generate path with ``prompt_logprobs`` (teacher forcing), so they
benefit from automatic prefix caching across a prompt's choices.

* ``/score_choices`` returns per-token logprobs + sum/mean/greedy for every
  choice, plus the best choice.
* ``/rank`` autoregressively selects an ordered list of ``k`` bundles from a
  shrinking candidate pool.
"""

import asyncio
import itertools
import time
from collections.abc import AsyncGenerator, Mapping

from fastapi import Request
from pydantic import Field

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.choice_scoring.async_batching import (
    rank_batch_pipelined_async,
    score_choices_batch_async,
)
from vllm.entrypoints.choice_scoring.fast_path import (
    extract_single_token_results,
    partition_single_token_groups,
    split_priming_wave,
)
from vllm.entrypoints.choice_scoring.inputs import (
    normalize_choices,
    normalize_prompt,
    validate_lengths,
)
from vllm.entrypoints.choice_scoring.native import (
    build_rank_output,
    build_score_output,
    make_native_rank_params,
    make_native_score_params,
)
from vllm.entrypoints.choice_scoring.params import (
    RankOutput,
    ScoreChoicesOutput,
)
from vllm.entrypoints.choice_scoring.reference import (
    extract_continuation_logprobs,
    make_scoring_sampling_params,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    OpenAIBaseModel,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.inputs import tokens_input
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import merge_async_iterators

logger = init_logger(__name__)

# Raw choice: text, pre-tokenized ids, or {"text"|"token_ids"}.
RawChoiceField = list[str] | list[list[int]] | list[dict]


# ============================================================================
# Protocol
# ============================================================================
class BatchScoreRequest(OpenAIBaseModel):
    model: str | None = None
    prompt: str | list[int] = Field(
        ..., description="Prompt text or pre-tokenized prompt token ids."
    )
    choices: RawChoiceField = Field(
        ...,
        description="Choices as text, token-id lists, or {text|token_ids} dicts.",
    )
    select_by: str = Field(
        default="mean",
        description="Metric for best_choice_index: 'mean' or 'sum'.",
    )
    num_prompt_logprobs: int = Field(
        default=0,
        description="prompt_logprobs count requested from the engine (>=0).",
    )
    add_special_tokens: bool = Field(
        default=True,
        description="Add special tokens (e.g. BOS) when tokenizing the prompt.",
    )
    return_tokens: bool = Field(
        default=True, description="Include decoded per-token strings in the output."
    )
    priority: int = Field(default=0)
    request_id: str = Field(default_factory=random_uuid)


class BatchRankRequest(OpenAIBaseModel):
    model: str | None = None
    prompt: str | list[int] = Field(...)
    candidates: RawChoiceField = Field(
        ..., description="Candidate bundles (text / token-ids / dicts)."
    )
    k: int = Field(..., description="Number of bundles to select (steps to stop).")
    select_by: str = Field(default="mean")
    num_prompt_logprobs: int = Field(default=0)
    add_special_tokens: bool = Field(default=True)
    return_tokens: bool = Field(default=True)
    priority: int = Field(default=0)
    request_id: str = Field(default_factory=random_uuid)


class ChoiceScoreResult(OpenAIBaseModel):
    index: int
    token_ids: list[int]
    token_logprobs: list[float]
    sum_logprob: float
    mean_logprob: float
    is_greedy: bool | None = None
    tokens: list[str] | None = None
    text: str | None = None


class BatchScoreResponse(OpenAIBaseModel):
    id: str = ""
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    prompt_token_ids: list[int]
    choices: list[ChoiceScoreResult]
    best_choice_index: int
    usage: UsageInfo


class RankStepResult(OpenAIBaseModel):
    order: int
    choice_index: int
    token_ids: list[int]
    token_logprobs: list[float]
    sum_logprob: float
    mean_logprob: float
    tokens: list[str] | None = None
    text: str | None = None


class BatchRankResponse(OpenAIBaseModel):
    id: str = ""
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    prompt_token_ids: list[int]
    selected: list[RankStepResult]
    truncated: bool
    usage: UsageInfo


# ============================================================================
# Serving
# ============================================================================
class ServingChoiceScoring(OpenAIServing):
    """Serving class for ``/score_choices`` and ``/rank``."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
        )

    def _tokenizer_fns(self):
        tokenizer = self.renderer.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available for choice scoring.")

        def _tok(text: str, add_special_tokens: bool) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)

        def _decode(token_ids):
            convert = getattr(tokenizer, "convert_ids_to_tokens", None)
            if convert is not None:
                return [str(t) for t in convert(list(token_ids))]
            return [tokenizer.decode([t]) for t in token_ids]

        return _tok, _decode

    def _make_async_score_batch_fn(
        self,
        request_id: str,
        num_prompt_logprobs: int,
        priority: int,
        lora_request,
        trace_headers,
        token_counter: list[int],
    ):
        # Each scoring call gets a distinct id space so concurrent rank
        # steps (pipelined driver) can never collide on request ids.
        call_counter = itertools.count()

        async def _generate_many(prompts_and_params, tag: str):
            generators: list[AsyncGenerator[RequestOutput, None]] = []
            for i, (prompt, sampling_params) in enumerate(prompts_and_params):
                generators.append(
                    self.engine_client.generate(
                        prompt,
                        sampling_params,
                        f"{request_id}-{tag}-{i}",
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=priority,
                    )
                )
            results: list[RequestOutput | None] = [None] * len(generators)
            async for i, res in merge_async_iterators(*generators):
                results[i] = res
            for i, res in enumerate(results):
                if res is None:
                    raise ValueError(f"no result for scoring request {tag}-{i}")
            return results

        async def _score_reference(pairs, tag: str):
            sequences = [list(ctx) + list(cand) for ctx, cand in pairs]
            token_counter[0] += sum(len(s) for s in sequences)
            prompts_and_params = [
                (
                    tokens_input(seq),
                    make_scoring_sampling_params(
                        num_prompt_logprobs, context_len=len(ctx)
                    ),
                )
                for (ctx, _), seq in zip(pairs, sequences)
            ]
            results = await _generate_many(prompts_and_params, tag)
            return [
                extract_continuation_logprobs(res.prompt_logprobs, len(ctx), cand)
                for (ctx, cand), res in zip(pairs, results)
            ]

        async def _score_batch(pairs):
            call_idx = next(call_counter)
            if not envs.VLLM_ENABLE_NATIVE_CHOICE_SCORING:
                return await _score_reference(pairs, str(call_idx))

            results: list = [None] * len(pairs)
            fast_groups: list = []
            slow_idx = list(range(len(pairs)))
            if num_prompt_logprobs == 0:
                fast_groups, slow_idx = partition_single_token_groups(pairs)
            if fast_groups:
                # One request per context; logprob_token_ids returns every
                # single-token candidate's logprob from a single forward.
                token_counter[0] += sum(
                    len(g.context) + len(g.cand_ids) for g in fast_groups
                )
                prompts_and_params = [
                    (
                        tokens_input(list(g.context)),
                        SamplingParams(
                            max_tokens=1,
                            temperature=0.0,
                            logprob_token_ids=list(g.cand_ids),
                            detokenize=False,
                        ),
                    )
                    for g in fast_groups
                ]
                outs = await _generate_many(prompts_and_params, f"{call_idx}-fast")
                for g, res in zip(fast_groups, outs):
                    completion = res.outputs[0]
                    extracted = extract_single_token_results(
                        g,
                        completion.token_ids[0],
                        completion.logprobs[0] if completion.logprobs else None,
                    )
                    for i, r in zip(g.pair_indices, extracted):
                        results[i] = r
            if slow_idx:
                # Context-priming: wave 1 caches each shared context once,
                # wave 2 siblings hit those blocks instead of re-prefilling.
                slow_pairs = [pairs[i] for i in slow_idx]
                for w, wave in enumerate(split_priming_wave(slow_pairs)):
                    if not wave:
                        continue
                    sub = [slow_pairs[j] for j in wave]
                    rs = await _score_reference(sub, f"{call_idx}-w{w}")
                    for j, r in zip(wave, rs):
                        results[slow_idx[j]] = r
            return results

        return _score_batch

    async def _prepare(self, request, raw_request, choices_field):
        error = await self._check_model(request)  # type: ignore[arg-type]
        if error is not None:
            return None, error
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            tok, decode = self._tokenizer_fns()
        except ValueError as e:
            return None, self.create_error_response(str(e))

        try:
            lora_request = self._maybe_get_adapters(request)  # type: ignore[arg-type]
            prompt_ids = normalize_prompt(
                request.prompt, tok, request.add_special_tokens
            )
            cands = normalize_choices(
                choices_field, tok, decode if request.return_tokens else None
            )
        except (ValueError, TypeError, RuntimeError) as e:
            logger.exception("Error preparing choice-scoring request")
            return None, self.create_error_response(str(e))

        ctx = {
            "lora_request": lora_request,
            "prompt_ids": prompt_ids,
            "cands": cands,
            "trace_headers": (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            ),
        }
        return ctx, None

    async def create_batch_score(
        self,
        request: BatchScoreRequest,
        raw_request: Request | None = None,
    ) -> BatchScoreResponse | ErrorResponse:
        ctx, error = await self._prepare(request, raw_request, request.choices)
        if error is not None:
            return error

        max_len = self.model_config.max_model_len
        try:
            validate_lengths(ctx["prompt_ids"], ctx["cands"], max_len)
        except ValueError as e:
            return self.create_error_response(str(e))

        base_id = self._base_request_id(raw_request, default=request.request_id)
        request_id = f"batch-score-{base_id}"
        token_counter = [0]

        if envs.VLLM_ENABLE_NATIVE_CHOICE_SCORING and request.num_prompt_logprobs == 0:
            # Engine-resident score: children scored and parsed in-core.
            try:
                output = await self._score_native_engine(request, ctx, request_id)
            except asyncio.CancelledError:
                return self.create_error_response("Client disconnected")
            except (ValueError, RuntimeError) as e:
                return self.create_error_response(str(e))
            token_counter[0] = len(ctx["prompt_ids"]) + sum(
                len(c.token_ids) for c in ctx["cands"]
            )
            return self._build_score_response(
                output, request, ctx["lora_request"], request_id, token_counter[0]
            )

        score_fn = self._make_async_score_batch_fn(
            request_id,
            request.num_prompt_logprobs,
            request.priority,
            ctx["lora_request"],
            ctx["trace_headers"],
            token_counter,
        )

        try:
            outputs = await score_choices_batch_async(
                [ctx["prompt_ids"]], [ctx["cands"]], score_fn, request.select_by
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))

        return self._build_score_response(
            outputs[0], request, ctx["lora_request"], request_id, token_counter[0]
        )

    async def create_batch_rank(
        self,
        request: BatchRankRequest,
        raw_request: Request | None = None,
    ) -> BatchRankResponse | ErrorResponse:
        ctx, error = await self._prepare(request, raw_request, request.candidates)
        if error is not None:
            return error
        if request.k < 0:
            return self.create_error_response("k must be >= 0")

        # Worst-case context growth: prompt + all selected bundles.
        lengths = sorted((len(c.token_ids) for c in ctx["cands"]), reverse=True)
        growth = sum(lengths[: max(0, min(request.k, len(lengths)) - 1)])
        try:
            validate_lengths(
                ctx["prompt_ids"], ctx["cands"], self.model_config.max_model_len, growth
            )
        except ValueError as e:
            return self.create_error_response(str(e))

        base_id = self._base_request_id(raw_request, default=request.request_id)
        request_id = f"batch-rank-{base_id}"
        token_counter = [0]

        if envs.VLLM_ENABLE_NATIVE_CHOICE_SCORING and request.k > 0:
            # Engine-resident rank: one request carries the whole pool; the
            # engine runs the entire k-step selection loop internally.
            try:
                output = await self._rank_native_engine(request, ctx, request_id)
            except asyncio.CancelledError:
                return self.create_error_response("Client disconnected")
            except (ValueError, RuntimeError) as e:
                return self.create_error_response(str(e))
            token_counter[0] = len(ctx["prompt_ids"]) + sum(
                len(c.token_ids) for c in ctx["cands"]
            )
            return self._build_rank_response(
                output, request, ctx["lora_request"], request_id, token_counter[0]
            )

        score_fn = self._make_async_score_batch_fn(
            request_id,
            request.num_prompt_logprobs,
            request.priority,
            ctx["lora_request"],
            ctx["trace_headers"],
            token_counter,
        )

        try:
            outputs = await rank_batch_pipelined_async(
                [ctx["prompt_ids"]],
                [ctx["cands"]],
                [request.k],
                score_fn,
                request.select_by,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))

        return self._build_rank_response(
            outputs[0], request, ctx["lora_request"], request_id, token_counter[0]
        )

    async def _score_native_engine(self, request, ctx, request_id: str):
        """Run one engine-resident score request and build its output."""
        params = make_native_score_params(ctx["cands"])
        final = None
        async for res in self.engine_client.generate(
            tokens_input(list(ctx["prompt_ids"])),
            params,
            request_id,
            lora_request=ctx["lora_request"],
            trace_headers=ctx["trace_headers"],
            priority=request.priority,
        ):
            final = res
        if final is None:
            raise RuntimeError("no output for engine-resident score request")
        return build_score_output(
            ctx["prompt_ids"],
            ctx["cands"],
            getattr(final, "choice_rank_result", None),
            request.select_by,
        )

    async def _rank_native_engine(self, request, ctx, request_id: str):
        """Run one engine-resident rank request and build its RankOutput."""
        params = make_native_rank_params(ctx["cands"], request.k, request.select_by)
        final = None
        async for res in self.engine_client.generate(
            tokens_input(list(ctx["prompt_ids"])),
            params,
            request_id,
            lora_request=ctx["lora_request"],
            trace_headers=ctx["trace_headers"],
            priority=request.priority,
        ):
            final = res
        if final is None:
            raise RuntimeError("no output for engine-resident rank request")
        return build_rank_output(
            ctx["prompt_ids"],
            ctx["cands"],
            getattr(final, "choice_rank_result", None),
        )

    def _build_score_response(
        self,
        out: ScoreChoicesOutput,
        request: BatchScoreRequest,
        lora_request,
        request_id: str,
        total_tokens: int,
    ) -> BatchScoreResponse:
        choices = [
            ChoiceScoreResult(
                index=c.index,
                token_ids=c.token_ids,
                token_logprobs=c.token_logprobs,
                sum_logprob=c.sum_logprob,
                mean_logprob=c.mean_logprob,
                is_greedy=c.is_greedy,
                tokens=c.tokens,
                text=c.text,
            )
            for c in out.choices
        ]
        return BatchScoreResponse(
            id=request_id,
            model=self.models.model_name(lora_request),
            prompt_token_ids=out.prompt_token_ids,
            choices=choices,
            best_choice_index=out.best_choice_index,
            usage=UsageInfo(
                prompt_tokens=len(out.prompt_token_ids),
                total_tokens=total_tokens,
                completion_tokens=0,
            ),
        )

    def _build_rank_response(
        self,
        out: RankOutput,
        request: BatchRankRequest,
        lora_request,
        request_id: str,
        total_tokens: int,
    ) -> BatchRankResponse:
        selected = [
            RankStepResult(
                order=s.order,
                choice_index=s.choice_index,
                token_ids=s.token_ids,
                token_logprobs=s.token_logprobs,
                sum_logprob=s.sum_logprob,
                mean_logprob=s.mean_logprob,
                tokens=s.tokens,
                text=s.text,
            )
            for s in out.selected
        ]
        return BatchRankResponse(
            id=request_id,
            model=self.models.model_name(lora_request),
            prompt_token_ids=out.prompt_token_ids,
            selected=selected,
            truncated=out.truncated,
            usage=UsageInfo(
                prompt_tokens=len(out.prompt_token_ids),
                total_tokens=total_tokens,
                completion_tokens=0,
            ),
        )

    async def _get_trace_headers(
        self,
        headers: Mapping[str, str],
    ) -> Mapping[str, str] | None:
        if not contains_trace_headers(headers):
            return None
        if not await self.engine_client.is_tracing_enabled():
            log_tracing_disabled_warning()
            return None
        return extract_trace_headers(headers)
