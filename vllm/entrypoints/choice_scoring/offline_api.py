# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline ``LLM`` API for choice scoring and ranking.

Provides :class:`ChoiceScoringOfflineMixin` with two methods mixed into
:class:`~vllm.entrypoints.llm.LLM`:

* ``batch_score(prompts, choices)`` -- teacher-forced per-token logprobs (plus
  sum/mean/greedy and the best choice) for every choice of every prompt.
* ``batch_rank(prompts, candidates, k)`` -- autoregressive ordered selection of
  ``k`` bundles per prompt from a shrinking candidate pool.

Both build on the model-independent, unit-tested layers (:mod:`core`,
:mod:`batching`, :mod:`reference`, :mod:`inputs`) and the existing generate
path with ``prompt_logprobs`` -- so they work on any causal LM and benefit from
automatic prefix caching across a prompt's choices.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.entrypoints.choice_scoring.batching import (
    rank_batch,
    score_choices_batch,
)
from vllm.entrypoints.choice_scoring.fast_path import (
    extract_single_token_results,
    partition_single_token_groups,
    split_priming_wave,
)
from vllm.entrypoints.choice_scoring.inputs import (
    RawChoice,
    RawPrompt,
    normalize_choices,
    normalize_prompt,
    validate_lengths,
)
from vllm.entrypoints.choice_scoring.params import (
    Candidate,
    RankOutput,
    ScoreChoicesOutput,
    SelectBy,
)
from vllm.entrypoints.choice_scoring.reference import (
    extract_continuation_logprobs,
    make_prompt_logprobs_score_batch_fn,
    make_scoring_sampling_params,
)
from vllm.entrypoints.choice_scoring.wavefront import (
    expected_rank_requests,
    rank_batch_wavefront,
)
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


def _is_single_prompt(prompts: object) -> bool:
    """A single prompt is a string or a flat sequence of ints."""
    if isinstance(prompts, str):
        return True
    if isinstance(prompts, Sequence):
        return all(isinstance(x, int) for x in prompts)
    return False


def _largest_growth(candidates: Sequence[Candidate], k: int) -> int:
    """Worst-case context growth before scoring a candidate during ranking.

    Before the final selection a prompt's context has grown by the previously
    selected bundles; bound that by the ``k-1`` longest candidates.
    """
    lengths = sorted((len(c.token_ids) for c in candidates), reverse=True)
    return sum(lengths[: max(0, min(k, len(lengths)) - 1)])


class ChoiceScoringOfflineMixin:
    """Mixin adding ``batch_score`` and ``batch_rank`` to ``LLM``."""

    # Provided by LLM.
    get_tokenizer: Callable[[], object]
    generate: Callable[..., list]
    model_config: object

    def _tokenize_fn(self):
        tokenizer = self.get_tokenizer()

        def _tok(text: str, add_special_tokens: bool) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)

        return _tok

    def _decode_tokens_fn(self):
        tokenizer = self.get_tokenizer()

        def _decode(token_ids):
            # Per-token surface strings for display.
            convert = getattr(tokenizer, "convert_ids_to_tokens", None)
            if convert is not None:
                return [str(t) for t in convert(list(token_ids))]
            return [tokenizer.decode([t]) for t in token_ids]

        return _decode

    def _make_score_batch_fn(
        self,
        num_prompt_logprobs: int,
        use_tqdm,
        lora_request: LoRARequest | None,
    ):
        def _generate_fn(
            sequences: list[list[int]],
            n_logprobs: int,
            context_lens: list[int],
        ):
            params = [
                make_scoring_sampling_params(n_logprobs, context_len=clen)
                for clen in context_lens
            ]
            prompts = [TokensPrompt(prompt_token_ids=list(seq)) for seq in sequences]
            return self.generate(
                prompts,
                params,
                use_tqdm=use_tqdm,
                lora_request=lora_request,
            )

        reference_fn = make_prompt_logprobs_score_batch_fn(
            _generate_fn, num_prompt_logprobs
        )

        def _fast_generate(groups):
            # One request per context: logprob_token_ids returns every
            # single-token candidate's logprob from one forward.
            prompts = [TokensPrompt(prompt_token_ids=list(g.context)) for g in groups]
            params = [
                SamplingParams(
                    max_tokens=1,
                    temperature=0.0,
                    logprob_token_ids=list(g.cand_ids),
                    detokenize=False,
                )
                for g in groups
            ]
            outs = self.generate(
                prompts, params, use_tqdm=use_tqdm, lora_request=lora_request
            )
            results = []
            for g, out in zip(groups, outs):
                completion = out.outputs[0]
                results.append(
                    extract_single_token_results(
                        g,
                        completion.token_ids[0],
                        completion.logprobs[0] if completion.logprobs else None,
                    )
                )
            return results

        def _score_batch(pairs):
            if not envs.VLLM_ENABLE_NATIVE_CHOICE_SCORING:
                return reference_fn(pairs)

            results: list = [None] * len(pairs)
            fast_groups: list = []
            slow_idx = list(range(len(pairs)))
            if num_prompt_logprobs == 0:
                fast_groups, slow_idx = partition_single_token_groups(pairs)
            if fast_groups:
                for g, rs in zip(fast_groups, _fast_generate(fast_groups)):
                    for i, r in zip(g.pair_indices, rs):
                        results[i] = r
            if slow_idx:
                # Context-priming: wave 1 caches each shared context once,
                # wave 2 siblings hit those blocks instead of re-prefilling.
                slow_pairs = [pairs[i] for i in slow_idx]
                for wave in split_priming_wave(slow_pairs):
                    if not wave:
                        continue
                    sub = [slow_pairs[j] for j in wave]
                    for j, r in zip(wave, reference_fn(sub)):
                        results[slow_idx[j]] = r
            return results

        return _score_batch

    def _prepare(
        self,
        prompts: RawPrompt | Sequence[RawPrompt],
        choices: Sequence[RawChoice] | Sequence[Sequence[RawChoice]],
        add_special_tokens: bool,
        return_tokens: bool,
    ) -> tuple[list[list[int]], list[list[Candidate]], bool]:
        single = _is_single_prompt(prompts)
        prompt_list = [prompts] if single else list(prompts)
        choice_lists = [choices] if single else list(choices)
        if len(prompt_list) != len(choice_lists):
            raise ValueError(
                f"{len(prompt_list)} prompts but {len(choice_lists)} choice "
                "lists; pass one choice list per prompt"
            )

        tok = self._tokenize_fn()
        decode = self._decode_tokens_fn() if return_tokens else None
        prompt_ids = [normalize_prompt(p, tok, add_special_tokens) for p in prompt_list]
        cand_lists = [normalize_choices(ch, tok, decode) for ch in choice_lists]
        return prompt_ids, cand_lists, single

    def batch_score(
        self,
        prompts: RawPrompt | Sequence[RawPrompt],
        choices: Sequence[RawChoice] | Sequence[Sequence[RawChoice]],
        *,
        select_by: SelectBy = "mean",
        num_prompt_logprobs: int = 0,
        add_special_tokens: bool = True,
        return_tokens: bool = True,
        use_tqdm: bool = True,
        lora_request: LoRARequest | None = None,
    ) -> ScoreChoicesOutput | list[ScoreChoicesOutput]:
        """Score choices for one prompt or a batch of prompts.

        Returns a single :class:`ScoreChoicesOutput` for a single prompt, or a
        list aligned with ``prompts`` for a batch.
        """
        max_len = getattr(self.model_config, "max_model_len", None)
        prompt_ids, cand_lists, single = self._prepare(
            prompts, choices, add_special_tokens, return_tokens
        )
        for pids, cands in zip(prompt_ids, cand_lists):
            validate_lengths(pids, cands, max_len)

        score_batch_fn = self._make_score_batch_fn(
            num_prompt_logprobs, use_tqdm, lora_request
        )
        outputs = score_choices_batch(prompt_ids, cand_lists, score_batch_fn, select_by)
        return outputs[0] if single else outputs

    def batch_rank(
        self,
        prompts: RawPrompt | Sequence[RawPrompt],
        candidates: Sequence[RawChoice] | Sequence[Sequence[RawChoice]],
        k: int | Sequence[int],
        *,
        select_by: SelectBy = "mean",
        num_prompt_logprobs: int = 0,
        add_special_tokens: bool = True,
        return_tokens: bool = True,
        use_tqdm: bool = True,
        lora_request: LoRARequest | None = None,
    ) -> RankOutput | list[RankOutput]:
        """Autoregressively select an ordered list of ``k`` bundles per prompt.

        Returns a single :class:`RankOutput` for a single prompt, or a list
        aligned with ``prompts`` for a batch.
        """
        max_len = getattr(self.model_config, "max_model_len", None)
        prompt_ids, cand_lists, single = self._prepare(
            prompts, candidates, add_special_tokens, return_tokens
        )
        n = len(prompt_ids)
        ks = [k] * n if isinstance(k, int) else list(k)
        if len(ks) != n:
            raise ValueError(f"expected {n} k values, got {len(ks)}")
        for pids, cands, ki in zip(prompt_ids, cand_lists, ks):
            validate_lengths(
                pids, cands, max_len, extra_growth=_largest_growth(cands, ki)
            )

        # All-single-token pools ride the logprob_token_ids fast path: each
        # rank step is then ONE request per prompt (instead of pool-size
        # teacher-forced sequences), so the lock-step driver with the fast
        # scorer beats the per-pair wavefront outright.
        all_single_token = (
            envs.VLLM_ENABLE_NATIVE_CHOICE_SCORING
            and num_prompt_logprobs == 0
            and all(
                len(c.token_ids) == 1 and c.token_ids[0] != 0
                for cands in cand_lists
                for c in cands
            )
            and all(len(cands) <= MAX_LOGPROB_TOKEN_IDS for cands in cand_lists)
        )
        engine = getattr(self, "llm_engine", None)
        if all_single_token:
            score_batch_fn = self._make_score_batch_fn(
                num_prompt_logprobs, use_tqdm, lora_request
            )
            outputs = rank_batch(prompt_ids, cand_lists, ks, score_batch_fn, select_by)
        elif (
            engine is not None
            and hasattr(engine, "add_request")
            and hasattr(engine, "step")
        ):
            # Pipelined wavefront: each prompt's next step is submitted the
            # moment its own previous step completes (no global step barrier).
            outputs = self._rank_wavefront(
                prompt_ids,
                cand_lists,
                ks,
                select_by,
                num_prompt_logprobs,
                use_tqdm,
                lora_request,
            )
        else:
            score_batch_fn = self._make_score_batch_fn(
                num_prompt_logprobs, use_tqdm, lora_request
            )
            outputs = rank_batch(prompt_ids, cand_lists, ks, score_batch_fn, select_by)
        return outputs[0] if single else outputs

    def _rank_wavefront(
        self,
        prompt_ids: list[list[int]],
        cand_lists: list[list[Candidate]],
        ks: list[int],
        select_by: SelectBy,
        num_prompt_logprobs: int,
        use_tqdm: bool,
        lora_request: LoRARequest | None,
    ) -> list[RankOutput]:
        """Drive :func:`rank_batch_wavefront` over the blocking engine."""
        engine = self.llm_engine  # type: ignore[attr-defined]
        prefix = f"cs-rank-{random_uuid()}"
        counter = itertools.count()
        # request_id -> (context_len, candidate_token_ids) for extraction.
        in_flight: dict[str, tuple[int, list[int]]] = {}

        def submit(ctx: list[int], cand: list[int]) -> str:
            request_id = f"{prefix}-{next(counter)}"
            params = make_scoring_sampling_params(
                num_prompt_logprobs, context_len=len(ctx)
            )
            engine.add_request(
                request_id,
                TokensPrompt(prompt_token_ids=list(ctx) + list(cand)),
                params,
                lora_request=lora_request,
            )
            in_flight[request_id] = (len(ctx), list(cand))
            return request_id

        def poll():
            outputs = engine.step()
            finished = []
            for out in outputs:
                if not getattr(out, "finished", False):
                    continue
                meta = in_flight.pop(out.request_id, None)
                if meta is None:
                    continue
                context_len, cand = meta
                finished.append(
                    (
                        out.request_id,
                        extract_continuation_logprobs(
                            out.prompt_logprobs, context_len, cand
                        ),
                    )
                )
            if not finished and in_flight and not engine.has_unfinished_requests():
                raise RuntimeError(
                    f"engine went idle with {len(in_flight)} outstanding "
                    "choice-scoring requests"
                )
            return finished

        def abort(request_ids: list[str]) -> None:
            with contextlib.suppress(Exception):
                engine.abort_request(request_ids)
            for request_id in request_ids:
                in_flight.pop(request_id, None)

        pbar = None
        progress_fn = None
        if use_tqdm:
            from tqdm.auto import tqdm

            total = sum(
                expected_rank_requests(len(cands), k)
                for cands, k in zip(cand_lists, ks)
            )
            pbar = tqdm(
                total=total,
                desc="Ranking (wavefront)",
                dynamic_ncols=True,
                unit="req",
            )
            progress_fn = pbar.update

        try:
            return rank_batch_wavefront(
                prompt_ids,
                cand_lists,
                ks,
                submit,
                poll,
                select_by,
                abort_fn=abort,
                progress_fn=progress_fn,
            )
        finally:
            if pbar is not None:
                pbar.close()
