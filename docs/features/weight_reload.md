# Hot Weight Reload

vLLM can reload model weights **in place**, without restarting the engine or the
server, for the same architecture, tokenizer and parallelism. This is intended
for the train-then-serve / RLHF iterate loop, where you repeatedly update
weights and keep serving.

After reloading, the [prefix cache](automatic_prefix_caching.md) is invalidated
so no KV state computed with the previous weights is reused.

## Offline: `LLM.reload_weights`

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen3-0.6B")          # in-process engine (TP=1)

# 1. From a checkpoint directory or HF id:
llm.reload_weights("/ckpts/step_1200")
llm.reload_weights("org/new-model")

# 2. Reload the engine's original model path (refresh):
llm.reload_weights()

# 3. From an in-memory state dict / live module — no disk, no extra copy:
llm.reload_weights(state_dict=hf_model)              # pass the nn.Module
llm.reload_weights(state_dict=hf_model.state_dict()) # or its state dict
```

### In-memory weights (no copy)

Passing `state_dict=` hands vLLM **references** to your live tensors — iterating
a `state_dict()` copies no weight data, and there is no disk round-trip. vLLM
then writes those tensors into its own pre-allocated, repacked (fused/sharded)
parameter buffers; that final write is intrinsic and is the only unavoidable
copy. Each worker's `load_weights` selects its own tensor-parallel shard, so
pass the full (unsharded) checkpoint-format state dict.

!!! warning "In-process vs. distributed"
    Tensors are passed by reference only with an **in-process** engine (TP=1, no
    multiprocessing). With multiprocessing/TP workers the tensors are serialized
    to each worker. For distributed GPU-to-GPU weight sync from a trainer,
    prefer the NCCL weight-transfer path (`--weight-transfer-backend nccl` and
    the `/v1/admin/rlhf/*` endpoints).

Set `is_checkpoint_format=False` only if the provided weights are already in
(sharded) kernel format rather than checkpoint format. Use
`reset_running_requests=False` to leave KV for in-flight requests untouched
(not recommended after a weight change).

## Online: `POST /reload_weights`

This endpoint is enabled only when the server runs in development mode
(`VLLM_SERVER_DEV_MODE=1`), because loading weights from an arbitrary path or
Hugging Face id is a privileged operation that must not be exposed publicly.

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B

curl -X POST http://localhost:8000/reload_weights \
  -H "Content-Type: application/json" \
  -d '{"weights_path": "/ckpts/step_1200"}'
```

Body fields (all optional): `weights_path` (local dir or HF id; omit to reload
the original path) and `reset_running_requests` (default `true`). Sending
in-memory tensors over HTTP is not supported — use the NCCL weight-transfer path
above for GPU-to-GPU sync from a trainer.

## Related

- [Sleep Mode](sleep_mode.md) — offload/discard weights and KV to free memory.
