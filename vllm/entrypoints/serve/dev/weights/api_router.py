# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dev endpoint to hot-reload model weights without relaunching the server.

Enabled only under ``VLLM_SERVER_DEV_MODE`` (set ``VLLM_SERVER_DEV_MODE=1``),
because loading weights from an arbitrary path/HF id is a privileged operation
and must not be exposed on a public server.

Reloads weights for the **same architecture** on every worker via
``collective_rpc("reload_weights")`` (the same primitive used by the offline
``LLM.reload_weights``), then invalidates the prefix cache so no KV computed
with the old weights is reused. Intended for the train-then-serve / RLHF loop
where model architecture, tokenizer and parallelism are unchanged.
"""

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/reload_weights")
async def reload_weights(raw_request: Request):
    """Reload model weights in place.

    Body (JSON, all optional):
        weights_path: local checkpoint dir or HF id. Omit to reload the
            engine's original model path.
        reset_running_requests: also invalidate KV for in-flight requests
            (default true).
    """
    try:
        body = await raw_request.json()
    except Exception:  # noqa: BLE001 - empty/invalid body -> defaults
        body = {}

    weights_path = body.get("weights_path")
    if weights_path is not None and not isinstance(weights_path, str):
        return JSONResponse(
            content={"error": "weights_path must be a string or omitted"},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    reset_running = bool(body.get("reset_running_requests", True))

    client = engine_client(raw_request)
    if client.errored:
        raise client.dead_error

    kwargs = {} if weights_path is None else {"weights_path": weights_path}
    logger.info("Hot-reloading weights (weights_path=%s)", weights_path)
    try:
        await client.collective_rpc("reload_weights", kwargs=kwargs)
    except Exception as e:  # noqa: BLE001 - surface load errors to the caller
        logger.exception("Weight reload failed")
        return JSONResponse(
            content={"status": "error", "detail": str(e)},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )

    await client.reset_prefix_cache(reset_running_requests=reset_running)
    return JSONResponse(content={"status": "ok", "weights_path": weights_path})


def attach_router(app: FastAPI):
    app.include_router(router)
