from __future__ import annotations

from fastapi import Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import (
    ClearQueueRequest,
    ClearQueueResponse,
    EnqueueTasksRequest,
    GraphBatchRequest,
    GraphLocateResponse,
    GraphVisualResponse,
    IDRequest,
    ListResultsResponse,
    QueueAckResponse,
    QueueStatusResponse,
    ResultDetailResponse,
    SubmitTasksRequest,
    SubmitTasksResponse,
)
from app.services import api_service
from app.services.service_errors import ServiceError


def _json_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message, "code": status_code})


def submit_tasks(
    request: SubmitTasksRequest, db: Session = Depends(get_db)
) -> SubmitTasksResponse | JSONResponse:
    try:
        return api_service.submit_tasks(request, db)
    except ServiceError as exc:
        return _json_error(exc.status_code, exc.message)


def enqueue_tasks(
    request: EnqueueTasksRequest, db: Session = Depends(get_db)
) -> QueueAckResponse | JSONResponse:
    try:
        return api_service.enqueue_tasks(request, db)
    except ServiceError as exc:
        return _json_error(exc.status_code, exc.message)


def get_task_status(db: Session = Depends(get_db)) -> QueueStatusResponse | JSONResponse:
    try:
        return api_service.get_task_status(db)
    except ServiceError as exc:
        return _json_error(exc.status_code, exc.message)


def get_graph_locate(
    limit: int | None = None, db: Session = Depends(get_db)
) -> GraphLocateResponse:
    return api_service.get_graph_locate(limit, db)


def clear_queue(request: ClearQueueRequest) -> ClearQueueResponse | JSONResponse:
    try:
        return api_service.clear_queue(request)
    except ServiceError as exc:
        return _json_error(exc.status_code, exc.message)


def list_results(
    page: int = 1, page_size: int = 20, db: Session = Depends(get_db)
) -> ListResultsResponse:
    return api_service.list_results(page, page_size, db)


def list_products(
    page: int = 1, page_size: int = 20, db: Session = Depends(get_db)
) -> ListResultsResponse:
    return api_service.list_products(page, page_size, db)


def get_result_detail(
    task_id: int, db: Session = Depends(get_db)
) -> ResultDetailResponse | JSONResponse:
    try:
        return api_service.get_result_detail(task_id, db)
    except ServiceError as exc:
        return _json_error(exc.status_code, exc.message)


def get_graph_view(
    task_id: int, db: Session = Depends(get_db)
) -> GraphVisualResponse | JSONResponse:
    try:
        return api_service.get_graph_view(task_id, db)
    except ServiceError as exc:
        return _json_error(exc.status_code, exc.message)


def build_graph(
    request: IDRequest, db: Session = Depends(get_db)
) -> QueueAckResponse | JSONResponse:
    try:
        return api_service.build_graph(request, db)
    except ServiceError as exc:
        return _json_error(exc.status_code, exc.message)


def build_graph_batch(
    request: GraphBatchRequest, db: Session = Depends(get_db)
) -> QueueAckResponse | JSONResponse:
    try:
        return api_service.build_graph_batch(request, db)
    except ServiceError as exc:
        return _json_error(exc.status_code, exc.message)


def get_preprocess_status() -> QueueStatusResponse:
    return api_service.get_preprocess_status()


def get_graph_status() -> QueueStatusResponse:
    return api_service.get_graph_status()
