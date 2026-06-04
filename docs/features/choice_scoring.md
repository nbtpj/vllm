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
the LM-head work per pair. Outputs are bit-for-bit semantically identical
(parity-tested against the default path); the flag is off by default until
you have run the parity tests on your hardware:

```bash
VLLM_ENABLE_NATIVE_CHOICE_SCORING=1 pytest tests/entrypoints/llm/test_choice_scoring.py -k native_window
```
