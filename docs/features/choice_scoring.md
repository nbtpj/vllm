# Choice Scoring and Ranking

vLLM can score and rank candidate continuations against a prompt using any
generative model (task `"generate"`). Both features share one primitive —
**teacher-forced continuation log-probabilities** — and benefit automatically
from [prefix caching](automatic_prefix_caching.md) across the choices of a
prompt.

- **`batch_score`** — given a prompt and a set of choices, return the per-token
  log-probability of every choice (plus `sum` / `mean` aggregates, a greedy
  flag, and the best choice). This is the multi-token generalization of the
  [Generative Scoring](../serving/online_serving/generative_scoring.md)
  endpoint, and resembles the `loglikelihood` request type used by
  multiple-choice evaluation harnesses.
- **`batch_rank`** — like generation, but each "move" is a multi-token *bundle*
  chosen from a per-prompt candidate pool. At each step every remaining
  candidate is re-scored against the current context (prompt + already-selected
  bundles), the highest average-log-prob candidate is selected and appended, and
  the pool shrinks. This repeats for `k` steps, producing an ordered list.

Both differ from the pooling-based
[Score API](../models/pooling_models/scoring.md), which uses cross-encoder /
bi-encoder models rather than a generative LM.

## Offline API

### `LLM.batch_score`

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen3-0.6B")

out = llm.batch_score(
    "The capital of France is",
    [" Paris", " London", " a large historic city"],
)
for ch in out.choices:
    print(ch.index, ch.sum_logprob, ch.mean_logprob, ch.is_greedy, ch.text)
print("best:", out.best_choice_index)
```

Each `ChoiceScore` contains `token_ids`, `token_logprobs`, `sum_logprob`,
`mean_logprob`, `is_greedy` (whether the choice was the argmax continuation at
every position), and optionally decoded `tokens`. `best_choice_index` is the
argmax by `select_by` (`"mean"` default, or `"sum"`), with lowest-index
tie-breaking.

Choices may be given as text **or** pre-tokenized id lists (or mixed). Pass a
list of prompts with a per-prompt list of choices for batched scoring:

```python
outs = llm.batch_score(
    ["Two plus two equals", "The opposite of hot is"],
    [[" four", " five", " seven"], [" cold", " warm"]],
)
```

!!! tip "Leading spaces matter"
    Whether a leading space belongs to a choice (`" Paris"` vs `"Paris"`) changes
    its tokenization for most tokenizers. Include the leading space when that is
    the intended continuation.

### `LLM.batch_rank`

```python
ranked = llm.batch_rank(
    "Rank these cities by relevance to France:",
    [" Paris", " Lyon", " Tokyo", " Marseille"],
    k=3,
)
for step in ranked.selected:           # ordered: step.order == 0 is the first pick
    print(step.order, step.choice_index, step.mean_logprob, step.text)
print("truncated:", ranked.truncated)  # True if k exceeded the pool size
```

`batch_rank` accepts a per-prompt `k` (`k=[2, 1]` for a batch). When `k` exceeds the
candidate pool, selection stops at pool exhaustion and `truncated` is `True`.

!!! note "Relationship between the two"
    The first `batch_rank` step scores candidates against the prompt alone, so
    `batch_rank(prompt, choices, k=1).selected[0].choice_index` always equals
    `batch_score(prompt, choices).best_choice_index` for the same `select_by`.

## Online serving

The same operations are exposed over HTTP as `POST /batch_score` and
`POST /batch_rank`. See
[Choice Scoring and Ranking (online)](../serving/online_serving/choice_scoring.md).

## Implementation notes

Scoring runs on the GPU via the engine's logprobs path: each `(context,
choice)` pair is teacher-forced (`prompt_logprobs`), and the shared prompt KV is
reused via prefix caching.

### Parallelism

- `batch_score` flattens every `(prompt, choice)` pair across the whole batch
  into a single fused engine call; the engine continuous-batches all pairs.
- `batch_rank` is pipelined **per prompt** (no global step barrier):
    - Offline, a wavefront driver feeds the engine directly
      (`add_request`/`step`); each prompt's next-step candidates are submitted
      the moment its own previous step finishes, so a prompt with a small pool
      never waits for a slower prompt in the same batch.
    - Online (and any `AsyncLLM` caller), every prompt runs its own async rank
      loop concurrently; the engine interleaves the in-flight scoring requests.
- A prompt's rank steps remain sequential by definition: step `t+1` is
  conditioned on the bundle selected at step `t`.

### Native scoring window (`VLLM_ENABLE_NATIVE_CHOICE_SCORING`)

By default the teacher-forcing path computes prompt logprobs at **every**
position of `context + choice`, which runs the LM head (a
`hidden_size x vocab_size` matmul plus a softmax over the vocab) once per
token of the full sequence. With `VLLM_ENABLE_NATIVE_CHOICE_SCORING=1`,
`batch_score`/`batch_rank` set `SamplingParams.prompt_logprobs_from` to the
context length, restricting LM-head computation to the **choice positions
only** -- for a 1000-token prompt with a 3-token choice that removes ~99% of
the LM-head work per pair.

The window also re-enables **prefix-cache reads**: plain `prompt_logprobs`
requests must skip reading cached prefix (cached positions produce no
logits), so by default every `(context, choice)` pair re-prefills its whole
context. With the window, the KV cache manager caps the cache hit just below
the window start, so the shared context KV is genuinely reused across a
prompt's choices and across rank steps.

With the flag on, `batch_rank` becomes **engine-resident** (general, any
bundle lengths): the client sends ONE request per prompt carrying the whole
candidate pool; the `EngineCore` intercepts it and runs the entire k-step
selection loop internally -- child scoring sequences share the context KV via
the prefix cache, their windowed logprob tensors are consumed and the best
bundle selected *in-core* (identical semantics to the client-side drivers,
parity-tested), and only the final structured result crosses the IPC
boundary. This removes per-candidate output serialization, client-side
logprob pythonization, and per-step submission round trips.

**Packed-pool scoring** (`VLLM_ENABLE_PACKED_CHOICE_SCORING=1`, requires
`VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` and the V1 model runner): the
coordinator packs all multi-token candidates of a prompt-step into ONE child
sequence scored in a single forward, using a branch attention mask
(candidates attend to the shared context and themselves only, with logical
rotary positions). Pool-size fewer scheduler requests per step; chunks of up
to 16 candidates; the packed suffix never enters the prefix cache. Huge
single-token pools (e.g. 30k variable candidates) shard automatically into
128-id `logprob_token_ids` children regardless of this flag.

!!! note "Measured guidance"
    Packed mode is parity-exact (fp32/bf16/4-bit verified) but
    throughput-neutral at small pools (~4 candidates): the per-step block-
    mask rebuild offsets the scheduler-request savings. Leave it off unless
    pools are large (it sustains ~2k scored candidates/s on 30k-candidate
    pools, where 16-candidate packs amortize the mask cost).

The flag enables two further request-level optimizations for `batch_score`:

- **Single-token fast path** -- when *all* of a prompt's choices are single
  tokens (the classic A/B/C/D case), no teacher forcing is needed: one
  request on the bare prompt with `SamplingParams.logprob_token_ids` returns
  every choice's logprob (and the argmax token, hence exact `is_greedy`)
  from a single forward. N choices collapse from N sequences to **1
  request** -- including every step of `batch_rank` over single-token pools.
  Falls back to teacher forcing when `num_prompt_logprobs > 0`, a choice is
  token id 0, or there are more than 128 choices.
- **Context-priming waves** -- multi-token choice groups are scored in two
  waves: wave 1 sends one pair per context (computing and caching the
  context blocks), wave 2 sends the siblings, which then hit the cache
  instead of re-prefilling the same context in parallel.

Outputs are semantically identical (parity-tested against the default path;
small fp deviations are possible in low-precision dtypes because the LM head
runs over a different batch shape). The flag is off by default until you have
run the parity tests on your hardware:

```bash
VLLM_ENABLE_NATIVE_CHOICE_SCORING=1 pytest tests/entrypoints/llm/test_choice_scoring.py -k native_window
```
