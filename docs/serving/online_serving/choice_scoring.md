# Choice Scoring and Ranking

Two endpoints score and rank candidate continuations against a prompt using a
generative model (task `"generate"`). They are the multi-token, multi-choice
counterparts of [Generative Scoring](generative_scoring.md); see the
[feature guide](../../features/choice_scoring.md) for the offline API and
semantics.

Both endpoints are available automatically when the server is started with a
generative model.

## `POST /score_choices`

Returns the teacher-forced per-token log-probabilities of every choice, with
`sum` / `mean` aggregates, a greedy flag, and the best choice.

Request fields: `prompt` (text or token ids), `choices` (list of text, token-id
lists, or `{text|token_ids}` dicts), `select_by` (`"mean"` default or `"sum"`),
`num_prompt_logprobs` (default `0`), `add_special_tokens` (default `true`),
`return_tokens` (default `true`).

```bash
curl -X POST http://localhost:8000/score_choices \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "choices": [" Paris", " London", " a large historic city"]
  }'
```

??? console "Response"

    ```json
    {
      "id": "score-choices-abc123",
      "object": "list",
      "model": "Qwen/Qwen3-0.6B",
      "prompt_token_ids": [791, 6864, 315, 9822, 374],
      "choices": [
        {
          "index": 0,
          "token_ids": [12366],
          "token_logprobs": [-1.84],
          "sum_logprob": -1.84,
          "mean_logprob": -1.84,
          "is_greedy": false,
          "tokens": ["ĠParis"],
          "text": " Paris"
        }
      ],
      "best_choice_index": 0,
      "usage": {"prompt_tokens": 5, "total_tokens": 18, "completion_tokens": 0}
    }
    ```

## `POST /rank`

Autoregressively selects an ordered list of `k` bundles from the candidate pool,
re-scoring the remaining candidates against the growing context at each step.

Request fields: `prompt`, `candidates` (same forms as `choices`), `k` (steps to
stop), plus the same `select_by` / `num_prompt_logprobs` / `add_special_tokens`
/ `return_tokens` options.

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Rank these cities by relevance to France:",
    "candidates": [" Paris", " Lyon", " Tokyo", " Marseille"],
    "k": 3
  }'
```

??? console "Response"

    ```json
    {
      "id": "rank-abc123",
      "object": "list",
      "model": "Qwen/Qwen3-0.6B",
      "prompt_token_ids": [...],
      "selected": [
        {"order": 0, "choice_index": 0, "token_ids": [12366],
         "token_logprobs": [-1.2], "sum_logprob": -1.2, "mean_logprob": -1.2,
         "text": " Paris"}
      ],
      "truncated": false,
      "usage": {"prompt_tokens": 9, "total_tokens": 42, "completion_tokens": 0}
    }
    ```

If `k` exceeds the number of candidates, selection stops when the pool is
exhausted and `truncated` is `true`. Empty `choices`/`candidates` or a negative
`k` return `400`.
