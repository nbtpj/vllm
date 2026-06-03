# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client examples for the /batch_score, /batch_rank and /reload_weights endpoints.

Start a server first (any generative model), e.g.:
    vllm serve distilbert/distilgpt2 --enforce-eager
    # add VLLM_SERVER_DEV_MODE=1 to enable /reload_weights:
    # VLLM_SERVER_DEV_MODE=1 vllm serve distilbert/distilgpt2 --enforce-eager

Then:
    python examples/online_serving/choice_scoring_client.py
    python examples/online_serving/choice_scoring_client.py --base-url http://host:8000
"""

import argparse

import requests


def batch_score(base_url: str) -> None:
    print("\n=== POST /batch_score ===")
    resp = requests.post(
        f"{base_url}/batch_score",
        json={
            "prompt": "The capital of France is",
            "choices": [" Paris", " London", " a large historic city"],
            "select_by": "mean",
        },
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    for ch in body["choices"]:
        print(f"  [{ch['index']}] mean={ch['mean_logprob']:.3f} "
              f"sum={ch['sum_logprob']:.3f} text={ch['text']!r}")
    print(f"  -> best_choice_index = {body['best_choice_index']}")


def score_with_token_ids(base_url: str) -> None:
    print("\n=== POST /batch_score (pre-tokenized choices) ===")
    resp = requests.post(
        f"{base_url}/batch_score",
        json={"prompt": "Two plus two equals", "choices": [[604], [642]]},
        timeout=60,
    )
    resp.raise_for_status()
    print(f"  best_choice_index = {resp.json()['best_choice_index']}")


def rank(base_url: str) -> None:
    print("\n=== POST /batch_rank ===")
    resp = requests.post(
        f"{base_url}/batch_rank",
        json={
            "prompt": "List European capital cities:",
            "candidates": [" Paris", " Berlin", " Tokyo", " Madrid"],
            "k": 3,
        },
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"  truncated={body['truncated']}")
    for step in body["selected"]:
        print(f"    #{step['order']}: choice {step['choice_index']} "
              f"(mean={step['mean_logprob']:.3f}) {step['text']!r}")


def reload_weights(base_url: str) -> None:
    print("\n=== POST /reload_weights (requires VLLM_SERVER_DEV_MODE=1) ===")
    resp = requests.post(
        f"{base_url}/reload_weights",
        json={},  # omit weights_path -> reload the original model path
        timeout=600,
    )
    if resp.status_code == 404:
        print("  endpoint not found: start the server with VLLM_SERVER_DEV_MODE=1")
        return
    print(f"  status={resp.status_code} body={resp.json()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--reload", action="store_true", help="also exercise /reload_weights"
    )
    args = parser.parse_args()

    batch_score(args.base_url)
    score_with_token_ids(args.base_url)
    rank(args.base_url)
    if args.reload:
        reload_weights(args.base_url)


if __name__ == "__main__":
    main()
