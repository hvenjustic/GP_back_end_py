from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.repositories.site_task_repository import get_task_by_id
from app.schemas import GraphBuildRequest, GraphBuildResponse
from app.services.crawl_tasks import build_graph_task

router = APIRouter()


@router.post("/graph/build", response_model=GraphBuildResponse, status_code=202)
def build_graph(
    request: GraphBuildRequest, db: Session = Depends(get_db)
) -> GraphBuildResponse:
    task = get_task_by_id(db, request.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="site_task not found")

    async_result = build_graph_task.delay(request.task_id)
    return GraphBuildResponse(
        task_id=request.task_id,
        status="QUEUED",
        celery_task_id=async_result.id,
    )
