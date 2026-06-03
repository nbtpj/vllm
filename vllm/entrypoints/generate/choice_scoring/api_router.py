# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.generate.choice_scoring.serving import (
    BatchRankRequest,
    BatchRankResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    ServingChoiceScoring,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger

router = APIRouter()

logger = init_logger(__name__)

_RESPONSES = {
    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
}


def _handler(request: Request) -> ServingChoiceScoring | None:
    return request.app.state.serving_choice_scoring


@router.post(
    "/batch_score",
    dependencies=[Depends(validate_json_request)],
    responses=_RESPONSES,
)
@with_cancellation
@load_aware_call
async def create_batch_score(request: BatchScoreRequest, raw_request: Request):
    handler = _handler(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support the batch_score API")
    result = await handler.create_batch_score(request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    assert isinstance(result, BatchScoreResponse)
    return JSONResponse(content=result.model_dump())


@router.post(
    "/batch_rank",
    dependencies=[Depends(validate_json_request)],
    responses=_RESPONSES,
)
@with_cancellation
@load_aware_call
async def create_batch_rank(request: BatchRankRequest, raw_request: Request):
    handler = _handler(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support the batch_rank API")
    result = await handler.create_batch_rank(request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    assert isinstance(result, BatchRankResponse)
    return JSONResponse(content=result.model_dump())


def register_choice_scoring_api_router(app: FastAPI):
    app.include_router(router)
