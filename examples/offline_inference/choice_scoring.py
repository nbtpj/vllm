# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comprehensive offline examples for choice scoring, ranking and weight reload.

Demonstrates:
  * score_choices  -- per-token logprobs + sum/mean/greedy + best choice
  * multiple-choice accuracy (lm-eval "loglikelihood" style)
  * batched scoring with per-prompt choice sets of different sizes
  * text vs pre-tokenized choices (and select_by mean vs sum)
  * rank          -- autoregressive ordered selection from a shrinking pool
  * reload_weights -- hot reload without relaunching (path / in-RAM state dict)

Run:
    python examples/offline_inference/choice_scoring.py
    python examples/offline_inference/choice_scoring.py --model Qwen/Qwen3-0.6B
"""

import argparse

from vllm import LLM


def demo_score_basic(llm: LLM) -> None:
    print("\n=== score_choices: basic ===")
    out = llm.score_choices(
        "The capital of France is",
        [" Paris", " London", " a large historic city"],
    )
    for ch in out.choices:
        toks = "".join(ch.tokens) if ch.tokens else ""
        print(
            f"  [{ch.index}] sum={ch.sum_logprob:7.3f} mean={ch.mean_logprob:7.3f} "
            f"greedy={ch.is_greedy!s:5} tokens={toks!r} text={ch.text!r}"
        )
    print(f"  -> best_choice_index = {out.best_choice_index}")


def demo_multiple_choice_accuracy(llm: LLM) -> None:
    """lm-eval 'loglikelihood' style: pick the choice with the best mean logprob."""
    print("\n=== score_choices: multiple-choice accuracy ===")
    questions = [
        ("2 + 2 = ", [" 4", " 5", " 22"], 0),
        ("The sky is ", [" blue", " green", " loud"], 0),
        ("The opposite of up is ", [" down", " sideways", " purple"], 0),
    ]
    prompts = [q for q, _, _ in questions]
    choices = [c for _, c, _ in questions]
    answers = [a for _, _, a in questions]

    outs = llm.score_choices(prompts, choices, select_by="mean")
    correct = 0
    for (q, ch, gold), out in zip(questions, outs):
        pred = out.best_choice_index
        correct += int(pred == gold)
        print(f"  {q!r:35} pred={ch[pred]!r:12} gold={ch[gold]!r:12} "
              f"{'OK' if pred == gold else 'X'}")
    print(f"  -> accuracy = {correct}/{len(questions)}")


def demo_batch_and_token_ids(llm: LLM) -> None:
    print("\n=== score_choices: batch + token-id choices + select_by ===")
    tok = llm.get_tokenizer()
    prompt = "Two plus two equals"
    text_choices = [" four", " five", " a number"]
    id_choices = [tok.encode(c, add_special_tokens=False) for c in text_choices]

    by_mean = llm.score_choices(prompt, text_choices, select_by="mean")
    by_sum = llm.score_choices(prompt, id_choices, select_by="sum")  # token-id input
    print(f"  best by mean = {by_mean.best_choice_index}")
    print(f"  best by sum  = {by_sum.best_choice_index}  (token-id input)")

    # A real batch: each prompt has its own (differently sized) choice set.
    outs = llm.score_choices(
        ["The opposite of hot is", "A baby dog is called a"],
        [[" cold", " warm"], [" puppy", " kitten", " calf"]],
    )
    print(f"  batch best indices = {[o.best_choice_index for o in outs]}")


def demo_rank(llm: LLM) -> None:
    print("\n=== rank: autoregressive ordered selection ===")
    prompt = "List European capital cities:"
    cands = [" Paris", " Berlin", " Tokyo", " Madrid", " Cairo"]

    ranked = llm.rank(prompt, cands, k=3)
    print(f"  top-3 (truncated={ranked.truncated}):")
    for step in ranked.selected:
        print(f"    #{step.order}: {cands[step.choice_index]!r} "
              f"(mean={step.mean_logprob:.3f})")

    # k > pool size -> full ordering, truncated=True.
    full = llm.rank(prompt, cands, k=99)
    order = [cands[s.choice_index] for s in full.selected]
    print(f"  full ordering (truncated={full.truncated}): {order}")

    # Batched rank with a per-prompt k.
    outs = llm.rank(
        ["Rank by size:", "Rank by heat:"],
        [[" elephant", " mouse", " whale"], [" sun", " ice", " fire"]],
        k=[2, 1],
    )
    print(f"  batch selection counts = {[len(o.selected) for o in outs]}")


def demo_reload_weights(llm: LLM) -> None:
    print("\n=== reload_weights ===")
    # Reload the original weights in place (no relaunch). For the RLHF loop you
    # would instead pass a new checkpoint or an in-memory state dict:
    #   llm.reload_weights("/ckpts/step_1200")        # path or HF id
    #   llm.reload_weights(state_dict=hf_model)        # live module, no copy
    #   llm.reload_weights(state_dict=hf_model.state_dict())
    llm.reload_weights()
    print("  reloaded original weights in place (prefix cache invalidated)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert/distilgpt2")
    args = parser.parse_args()

    llm = LLM(model=args.model, enforce_eager=True)
    demo_score_basic(llm)
    demo_multiple_choice_accuracy(llm)
    demo_batch_and_token_ids(llm)
    demo_rank(llm)
    demo_reload_weights(llm)


if __name__ == "__main__":
    main()
