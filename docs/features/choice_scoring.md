# Choice Scoring and Ranking

vLLM can score and rank candidate continuations against a prompt using any
generative model (task `"generate"`). Both features share one primitive —
**teacher-forced continuation log-probabilities** — and benefit automatically
from [prefix caching](automatic_prefix_caching.md) across the choices of a
prompt.

- **`score`** — given a prompt and a set of choices, return the per-token
  log-probability of every choice (plus `sum` / `mean` aggregates, a greedy
  flag, and the best choice). This is the multi-token generalization of the
  [Generative Scoring](../serving/online_serving/generative_scoring.md)
  endpoint, and resembles the `loglikelihood` request type used by
  multiple-choice evaluation harnesses.
- **`rank`** — like generation, but each "move" is a multi-token *bundle*
  chosen from a per-prompt candidate pool. At each step every remaining
  candidate is re-scored against the current context (prompt + already-selected
  bundles), the highest average-log-prob candidate is selected and appended, and
  the pool shrinks. This repeats for `k` steps, producing an ordered list.

Both differ from the pooling-based
[Score API](../models/pooling_models/scoring.md), which uses cross-encoder /
bi-encoder models rather than a generative LM.

## Offline API

### `LLM.score_choices`

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen3-0.6B")

out = llm.score_choices(
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
outs = llm.score_choices(
    ["Two plus two equals", "The opposite of hot is"],
    [[" four", " five", " seven"], [" cold", " warm"]],
)
```

!!! tip "Leading spaces matter"
    Whether a leading space belongs to a choice (`" Paris"` vs `"Paris"`) changes
    its tokenization for most tokenizers. Include the leading space when that is
    the intended continuation.

### `LLM.rank`

```python
ranked = llm.rank(
    "Rank these cities by relevance to France:",
    [" Paris", " Lyon", " Tokyo", " Marseille"],
    k=3,
)
for step in ranked.selected:           # ordered: step.order == 0 is the first pick
    print(step.order, step.choice_index, step.mean_logprob, step.text)
print("truncated:", ranked.truncated)  # True if k exceeded the pool size
```

`rank` accepts a per-prompt `k` (`k=[2, 1]` for a batch). When `k` exceeds the
candidate pool, selection stops at pool exhaustion and `truncated` is `True`.

!!! note "Relationship between the two"
    The first `rank` step scores candidates against the prompt alone, so
    `rank(prompt, choices, k=1).selected[0].choice_index` always equals
    `score_choices(prompt, choices).best_choice_index` for the same `select_by`.

## Online serving

The same operations are exposed over HTTP as `POST /score_choices` and
`POST /rank`. See
[Choice Scoring and Ranking (online)](../serving/online_serving/choice_scoring.md).

## Implementation notes

Scoring runs on the GPU via the engine's logprobs path: each `(context,
choice)` pair is teacher-forced (`prompt_logprobs`), and the shared prompt KV is
reused via prefix caching. An optional native single-round-trip backend (gated
by `VLLM_ENABLE_NATIVE_CHOICE_SCORING`) keeps the whole `rank` loop on the
worker; it is validated for parity against the logprobs path.
