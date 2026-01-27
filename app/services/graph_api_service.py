from __future__ import annotations

from sqlalchemy.orm import Session

from app.repositories.site_task_repository import get_task_by_id
from app.schemas import GraphBuildRequest, GraphBuildResponse
from app.services.crawl_tasks import build_graph_task
from app.services.queue_keys import GRAPH_ACTIVE_SET_KEY, GRAPH_TASK_MAP_KEY
from app.services.redis_client import get_redis_client
from app.services.service_errors import ServiceError


def build_graph(request: GraphBuildRequest, db: Session) -> GraphBuildResponse:
    task = get_task_by_id(db, request.task_id)
    if not task:
        raise ServiceError(status_code=404, message="site_task not found")

    async_result = build_graph_task.delay(request.task_id)
    rdb = get_redis_client()
    rdb.sadd(GRAPH_ACTIVE_SET_KEY, async_result.id)
    rdb.hset(GRAPH_TASK_MAP_KEY, async_result.id, int(request.task_id))
    return GraphBuildResponse(
        task_id=request.task_id,
        status="QUEUED",
        celery_task_id=async_result.id,
    )
