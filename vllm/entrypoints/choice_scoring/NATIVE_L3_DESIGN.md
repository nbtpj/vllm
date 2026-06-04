# Native (L3) choice-scoring & ranking — engine wiring blueprint

> **Status update.** Option (a) below is **implemented**: the
> `SamplingParams.prompt_logprobs_from` window (gated by
> `VLLM_ENABLE_NATIVE_CHOICE_SCORING`) restricts LM-head computation to the
> candidate positions in both V1 (`gpu_model_runner._get_prompt_logprobs_dict`)
> and V2 (`gpu/sample/prompt_logprob.PromptLogprobsWorker`) model runners, with
> `None`-padding in the output processor and GPU parity tests
> (`tests/entrypoints/llm/test_choice_scoring.py -k native_window`).
> Cross-prompt pipelining is also implemented: `wavefront.rank_batch_wavefront`
> (offline, engine `add_request`/`step`) and
> `async_batching.rank_batch_pipelined_async` (async/serving) remove the
> global rank step barrier. Option (b) — the resident single-request loop with
> on-device selection via `native_tensor_ops` — remains future work.

This document specifies the remaining **GPU-only** work to make `score` and
`rank` execute natively inside the V1 worker (single round-trip, on-device
selection), instead of the reference `prompt_logprobs` orchestration. The
model-independent semantics, the batched orchestration, and the **device-side
scoring/selection kernel** (`native_tensor_ops.py`) are already implemented and
unit-tested on CPU; this is the plumbing that feeds that kernel from the engine.

It is a blueprint because it touches the scheduler / model-runner / KV cache —
code that can only be validated on a GPU. Every step lists the exact file and
the existing primitive to reuse (from the spec-decode verify-and-accept path).

## Principle

Both features reduce to **teacher-forced scoring of N prefix-sharing candidate
sequences in one forward**, plus, for `rank`, **accepting the best bundle into
the KV context and re-scoring the shrunken pool**. This is structurally
identical to speculative-decoding *verify* (score proposed tokens in one pass)
and *accept* (commit accepted tokens to KV). We reuse that machinery rather than
inventing a new attention backend.

Candidates that share a prefix are laid out as **separate sequences** that share
the prefix via automatic prefix caching (robust, backend-agnostic) — NOT as one
causal chain (candidate 2 must not attend to candidate 1). A single-forward
branched/tree mask is a later optimization for backends that support it.

## Components already done (CPU-tested)

- `params.py`, `core.py` — semantics (38 tests)
- `batching.py`, `async_batching.py` — cross-prompt + per-step batching (21 tests)
- `reference.py` — `prompt_logprobs` oracle (11 tests)
- `inputs.py` — text/token-id/dict normalization (21 tests)
- `native_tensor_ops.py` — on-device `token_logprobs_and_greedy`,
  `aggregate_ragged` (CSR/ragged sum·mean·greedy), `select_best`
  (masked argmax, lowest-index tie-break), `score_candidates` (9 parity tests)

## Wiring to implement (GPU)

### 1. Request params — `vllm/sampling_params.py` (or a new struct)
Add a `msgspec.Struct` `ChoiceScoringParams`:
```
mode: Literal["score", "rank"]
candidate_token_ids: list[list[int]]   # per-prompt pool (front-end tokenized)
k: int = 1                             # rank only
select_by: Literal["mean", "sum"] = "mean"
num_logprobs: int = 0                  # extra top-k to also return, optional
```
Front-end tokenization/validation already exists in `inputs.py` — reuse it in
the Processor.

### 2. Carry it on the request
- `vllm/v1/engine/__init__.py::EngineCoreRequest` — add
  `choice_scoring_params: ChoiceScoringParams | None` (mutually exclusive with
  sampling/pooling params; extend the `params` property).
- `vllm/v1/engine/input_processor.py::process_inputs` (~line 313) — add an
  `elif isinstance(params, ChoiceScoringParams)` branch; the "prompt" is the
  shared context; `max_tokens` is irrelevant (no free generation).
- `vllm/v1/request.py::Request` — store the params; treat like pooling for
  lifetime (finishes when its result is produced); for `rank` it stays resident
  across `k` internal steps (see §4).

### 3. Scheduling
Gate behind `VLLM_ENABLE_NATIVE_CHOICE_SCORING`. Two options:

**(a) Multi-request (simplest, recommended first).** The scheduler does nothing
special: the front-end expands a `score` request into N candidate
sub-sequences (context+cand_i) and submits them; the model runner returns their
candidate-position logprobs; `batching.score_choices_batch` aggregates. `rank`
is driven by `async_batching.rank_batch_async` exactly as the reference does,
but each candidate forward uses path (b)'s gather instead of full
`prompt_logprobs`. This already works via the reference path — the only "native"
win here is requesting *only* the candidate-position logits.

**(b) Resident single-request (the true L3).** One `EngineCoreRequest` carries
the whole pool; the scheduler keeps it resident and re-invokes the worker for
`k` steps. Mirror how spec-decode advances `num_computed_tokens` across steps
(`scheduler.py` ~299–346; `gpu_model_runner` state in `requests`/`InputBatch`).

### 4. Model runner — `vllm/v1/worker/gpu_model_runner.py`
Add `_score_choices(...)` alongside `_pool` (~3329) / `sample_tokens` (~4381):
1. Build `logits_indices` for **candidate positions only** (reuse the
   `_calc_spec_decode_metadata` index math, ~2722–2800) — we need the logit that
   *predicts* each candidate token, i.e. position `ctx_len-1 .. ctx_len+L-2`.
2. `logits = compute_logits(hidden_states[logits_indices])`.
3. Build `target_token_ids` (the candidate tokens) and `candidate_offsets`
   (CSR) and call `native_tensor_ops.score_candidates(...)`.
4. **score:** return per-candidate `sum/mean/is_greedy` (+ ragged
   `token_logprobs`) in `ModelRunnerOutput`.
5. **rank (resident):** `best = select_best(metric, valid_mask)`; mark that
   candidate selected in `valid_mask`; **accept its tokens into KV** by
   advancing the context (same mechanism as accepting draft tokens — commit the
   block table and bump `num_computed_tokens`, ~1498–1547, 1888); loop to step 1
   for the next step. Stop after `k` steps or pool exhaustion. Emit the ordered
   list.

Attention: candidates are separate sequences sharing the prefix KV (prefix
cache). Accepted bundles extend the shared context; rejected candidates' KV is
simply not extended. No custom mask needed for option (a)/(b)-as-separate-seqs.

### 5. Output plumbing
- `ModelRunnerOutput` / `EngineCoreOutput` — add a `choice_scoring_output`
  field (parallels `pooling_output`); `scheduler.update_from_outputs`
  (~1435–1552) finishes the request when present.
- `OutputProcessor` — build `ScoreChoicesOutput` / `RankOutput` (the existing
  dataclasses) from the device tensors.

### 6. Surfacing
`LLM.score_choices/.rank` and the HTTP handlers pick the native backend when the
flag is on and the model/runner supports it, else fall back to the reference
path. The public API and output types are unchanged.

## Validation plan (the safety net)

Parity tests (`tests/.../test_choice_scoring_native_parity.py`, GPU): for the
same model + inputs, assert the native backend's `ScoreChoicesOutput` /
`RankOutput` equal the reference path within fp tolerance (`token_logprobs`
abs≤1e-3; identical `best_choice_index` / selection order / `truncated`). The
reference path is the oracle; the native path may not ship until parity is green.

## Env flag
`VLLM_ENABLE_NATIVE_CHOICE_SCORING` (add to `vllm/envs.py`), default off. While
off, everything uses the reference path (already correct and tested).
