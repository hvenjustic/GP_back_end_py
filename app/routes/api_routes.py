from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import CrawlJob, SitePage, SiteTask
from app.repositories.crawl_job_repository import add_job, get_latest_job_by_root_url
from app.repositories.site_page_repository import add_page
from app.repositories.site_task_repository import (
    get_task_by_id,
    get_tasks_by_ids,
    list_geo_locations,
    list_tasks,
    upsert_task_for_submission,
)
from app.schemas import (
    ClearQueueRequest,
    ClearQueueResponse,
    CrawlJobMeta,
    CrawlRequest,
    EnqueueTasksRequest,
    GraphLocateResponse,
    GraphVisualResponse,
    IDRequest,
    ListResultsResponse,
    QueueAckResponse,
    QueueStatusResponse,
    ResultDetailResponse,
    ResultItem,
    SubmitTasksRequest,
    SubmitTasksResponse,
)
from app.services.crawl_service import build_crawl_params, hash_url, normalize_url
from app.services.crawl_tasks import build_graph_task, celery_app, crawl_job_task
from app.services.graph_service import is_crawl_job_done
from app.services.redis_client import get_redis_client

router = APIRouter(prefix="/api")

CRAWL_QUEUE_KEY = "crawl4ai:queue"
CRAWL_ACTIVE_SET_KEY = "crawl4ai:active"
PREPROCESS_QUEUE_KEY = "kg:preprocess:queue"
PREPROCESS_ACTIVE_SET_KEY = "kg:preprocess:active"
GRAPH_QUEUE_KEY = "kg:graph:queue"
GRAPH_ACTIVE_SET_KEY = "kg:graph:active"

INVALID_ID_CHARS = re.compile(r"[^a-zA-Z0-9:_-]+")


def _json_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message, "code": status_code})


def _derive_site_name(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    return parsed.netloc.strip() if parsed.netloc else ""


def _is_valid_url(raw_url: str) -> bool:
    parsed = urlparse(raw_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _queue_pending(rdb, queue_key: str, active_key: str) -> int:
    return int(rdb.llen(queue_key) + rdb.scard(active_key))


def _cleanup_crawl_active(db: Session, rdb) -> None:
    job_ids = list(rdb.smembers(CRAWL_ACTIVE_SET_KEY))
    if not job_ids:
        return
    rows = (
        db.query(CrawlJob.job_id, CrawlJob.status)
        .filter(CrawlJob.job_id.in_(job_ids))
        .all()
    )
    known_ids = {job_id for job_id, _status in rows}
    done_ids = {job_id for job_id, status in rows if is_crawl_job_done(status)}
    stale_ids = set(job_ids) - known_ids
    remove_ids = done_ids | stale_ids
    if remove_ids:
        rdb.srem(CRAWL_ACTIVE_SET_KEY, *remove_ids)


def _cleanup_celery_active(rdb, active_key: str) -> None:
    task_ids = list(rdb.smembers(active_key))
    if not task_ids:
        return
    done_ids: list[str] = []
    for task_id in task_ids:
        if celery_app.AsyncResult(task_id).ready():
            done_ids.append(task_id)
    if done_ids:
        rdb.srem(active_key, *done_ids)


def _build_crawl_job(
    db: Session, root_url: str, settings
) -> tuple[str, str] | None:
    params = build_crawl_params(settings, CrawlRequest(root_url=root_url))
    normalized_root = normalize_url(root_url, root_url, params)
    if not normalized_root:
        return None

    job_id = str(uuid4())
    now = datetime.utcnow()

    job = CrawlJob(
        job_id=job_id,
        root_url=normalized_root,
        status="PENDING",
        created_at=now,
        updated_at=now,
        discovered_count=1,
        queued_count=1,
        params=params.to_dict(),
    )
    add_job(db, job)

    root_page = SitePage(
        job_id=job_id,
        url=normalized_root,
        url_hash=hash_url(normalized_root),
        childrens=[],
        parent_url=None,
        depth=0,
        crawled=False,
        crawl_status="PENDING",
    )
    add_page(db, root_page)
    return job_id, normalized_root


def _build_crawl_job_meta(job: CrawlJob) -> CrawlJobMeta:
    return CrawlJobMeta(
        job_id=job.job_id,
        status=job.status,
        discovered_count=job.discovered_count,
        queued_count=job.queued_count,
        crawled_count=job.crawled_count,
        failed_count=job.failed_count,
        current_depth=job.current_depth,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        updated_at=job.updated_at,
        finished_at=job.finished_at,
    )


def _build_result_item(task: SiteTask, job: CrawlJob | None) -> ResultItem:
    item = ResultItem(
        id=int(task.id),
        name=task.name,
        site_name=task.site_name,
        url=task.url,
        crawled_at=task.crawled_at,
        llm_processed_at=task.llm_processed_at,
        is_crawled=bool(task.is_crawled),
        crawl_count=int(task.crawl_count or 0),
        page_count=int(task.page_count or 0),
        graph_json=task.graph_json,
        crawl_duration_ms=int(task.crawl_duration_ms or 0),
        llm_duration_ms=int(task.llm_duration_ms or 0),
        created_at=task.created_at,
        updated_at=task.updated_at,
        crawl_job=_build_crawl_job_meta(job) if job else None,
    )

    if job and is_crawl_job_done(job.status):
        item.is_crawled = True
        if item.crawled_at is None and job.finished_at is not None:
            item.crawled_at = job.finished_at
        if item.page_count == 0 and job.crawled_count > 0:
            item.page_count = job.crawled_count

    return item


def _parse_geo_location(raw: Any) -> tuple[float, float] | None:
    def parse_from_dict(value: dict[str, Any]) -> tuple[float, float] | None:
        lat = value.get("lat") or value.get("latitude")
        lng = value.get("lng") or value.get("longitude") or value.get("lon")
        if lat is None or lng is None:
            location = value.get("location")
            if isinstance(location, str):
                parts = [p.strip() for p in location.split(",")]
                if len(parts) == 2:
                    try:
                        lng = float(parts[0])
                        lat = float(parts[1])
                    except ValueError:
                        return None
        try:
            lat_f = float(lat)
            lng_f = float(lng)
        except (TypeError, ValueError):
            return None
        if lat_f == 0 or lng_f == 0:
            return None
        return lat_f, lng_f

    if raw is None:
        return None
    if isinstance(raw, dict):
        return parse_from_dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                return parse_from_dict(obj)
        if "," in text:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) == 2:
                try:
                    lng = float(parts[0])
                    lat = float(parts[1])
                except ValueError:
                    return None
                if lat == 0 or lng == 0:
                    return None
                return lat, lng
    return None


def _build_visual_elements(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities = payload.get("entities") if isinstance(payload, dict) else None
    relations = payload.get("relations") if isinstance(payload, dict) else None
    entities = entities if isinstance(entities, list) else []
    relations = relations if isinstance(relations, list) else []

    used_node_ids: set[str] = set()
    name_index: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []

    def gen_id(base: str) -> str:
        clean = INVALID_ID_CHARS.sub("_", base.strip())
        if not clean:
            clean = "node"
        original = clean
        suffix = 1
        while clean in used_node_ids:
            clean = f"{original}_{suffix}"
            suffix += 1
        used_node_ids.add(clean)
        return clean

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = str(ent.get("name") or "").strip()
        typ = str(ent.get("type") or "").strip()
        label = name or "未知实体"
        node_id = gen_id(f"{typ}:{name}")
        if name:
            name_index[name.lower()] = node_id
        nodes.append(
            {
                "id": node_id,
                "name": name,
                "type": typ,
                "label": label,
                "description": ent.get("description") or "",
                "extra": ent.get("extra") if isinstance(ent.get("extra"), dict) else {},
                "raw": {
                    "aliases": ent.get("aliases") or [],
                    "extra": ent.get("extra") if isinstance(ent.get("extra"), dict) else {},
                },
            }
        )

    used_edge_ids: dict[str, int] = {}
    edges: list[dict[str, Any]] = []
    for idx, rel in enumerate(relations):
        if not isinstance(rel, dict):
            continue
        src_name = str(rel.get("source") or "").strip()
        dst_name = str(rel.get("target") or "").strip()
        if not src_name or not dst_name:
            continue
        src_id = name_index.get(src_name.lower())
        dst_id = name_index.get(dst_name.lower())
        if not src_id or not dst_id:
            continue

        edge_type = str(rel.get("type") or "").strip() or "RELATED_TO"
        base_id = f"{src_id}:{edge_type}:{dst_id}"
        count = used_edge_ids.get(base_id, 0)
        edge_id = base_id if count == 0 else f"{base_id}_{count}"
        used_edge_ids[base_id] = count + 1
        edges.append(
            {
                "id": edge_id,
                "source": src_id,
                "target": dst_id,
                "type": edge_type,
                "label": edge_type,
                "raw": {
                    "index": idx,
                    "source_name": src_name,
                    "target_name": dst_name,
                },
            }
        )

    return nodes, edges


@router.post("/tasks", response_model=SubmitTasksResponse)
def submit_tasks(
    request: SubmitTasksRequest, db: Session = Depends(get_db)
) -> SubmitTasksResponse | JSONResponse:
    if not request.urls:
        return _json_error(400, "urls empty")

    created = 0
    for item in request.urls:
        raw_url = (item.url or "").strip()
        if not raw_url or not _is_valid_url(raw_url):
            continue

        name = (item.name or "").strip() or (item.site_name or "").strip()
        if not name:
            name = _derive_site_name(raw_url)

        upsert_task_for_submission(db, raw_url, name)
        created += 1

    db.commit()
    return SubmitTasksResponse(created=created)


@router.post("/tasks/crawl", response_model=QueueAckResponse)
def enqueue_tasks(
    request: EnqueueTasksRequest, db: Session = Depends(get_db)
) -> QueueAckResponse | JSONResponse:
    ids = [int(item) for item in request.ids if int(item) > 0]
    if not ids:
        return _json_error(400, "ids empty")

    settings = get_settings()
    tasks = get_tasks_by_ids(db, ids)
    if not tasks:
        return QueueAckResponse(queued=0, queue_key=CRAWL_QUEUE_KEY, pending=0)

    rdb = get_redis_client()
    queued = 0
    for task in tasks:
        raw_url = (task.url or "").strip()
        if not raw_url:
            continue
        try:
            result = _build_crawl_job(db, raw_url, settings)
            if result is None:
                continue
            job_id, _normalized = result
            db.commit()
        except Exception:
            db.rollback()
            continue

        crawl_job_task.delay(job_id)
        rdb.sadd(CRAWL_ACTIVE_SET_KEY, job_id)
        queued += 1

    _cleanup_crawl_active(db, rdb)
    pending = _queue_pending(rdb, CRAWL_QUEUE_KEY, CRAWL_ACTIVE_SET_KEY)
    return QueueAckResponse(queued=queued, queue_key=CRAWL_QUEUE_KEY, pending=pending)


@router.get("/tasks/status", response_model=QueueStatusResponse)
def get_task_status(db: Session = Depends(get_db)) -> QueueStatusResponse | JSONResponse:
    rdb = get_redis_client()
    _cleanup_crawl_active(db, rdb)
    pending = _queue_pending(rdb, CRAWL_QUEUE_KEY, CRAWL_ACTIVE_SET_KEY)
    return QueueStatusResponse(pending=pending, queue_key=CRAWL_QUEUE_KEY)


@router.get("/graph_locate", response_model=GraphLocateResponse)
def get_graph_locate(
    limit: int | None = None, db: Session = Depends(get_db)
) -> GraphLocateResponse:
    tasks = list_geo_locations(db, limit or 0)
    items: list[dict[str, Any]] = []
    for task in tasks:
        coords = _parse_geo_location(task.geo_location)
        if not coords:
            continue
        lat, lng = coords
        items.append({"id": int(task.id), "latitude": lat, "longitude": lng})
    return GraphLocateResponse(items=items, total=len(items))


@router.post("/queues/clear", response_model=ClearQueueResponse)
def clear_queue(request: ClearQueueRequest) -> ClearQueueResponse | JSONResponse:
    queue_name = (request.queue_name or "").strip()
    if not queue_name:
        return _json_error(400, "queue_name empty")
    rdb = get_redis_client()
    removed = int(rdb.delete(queue_name))
    return ClearQueueResponse(queue_name=queue_name, removed_keys=removed)


@router.get("/results", response_model=ListResultsResponse)
def list_results(
    page: int = 1, page_size: int = 20, db: Session = Depends(get_db)
) -> ListResultsResponse:
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    if page_size > 100:
        page_size = 100

    tasks, total = list_tasks(db, page, page_size)
    items: list[ResultItem] = []
    for task in tasks:
        job = get_latest_job_by_root_url(db, task.url)
        items.append(_build_result_item(task, job))
    return ListResultsResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/results/{task_id}", response_model=ResultDetailResponse)
def get_result_detail(
    task_id: int, db: Session = Depends(get_db)
) -> ResultDetailResponse | JSONResponse:
    if task_id <= 0:
        return _json_error(400, "id invalid")
    task = get_task_by_id(db, task_id)
    if not task:
        return _json_error(404, "task not found")
    job = get_latest_job_by_root_url(db, task.url)
    return ResultDetailResponse(item=_build_result_item(task, job))


@router.get("/results/{task_id}/graph_view", response_model=GraphVisualResponse)
def get_graph_view(
    task_id: int, db: Session = Depends(get_db)
) -> GraphVisualResponse | JSONResponse:
    if task_id <= 0:
        return _json_error(400, "id invalid")
    task = get_task_by_id(db, task_id)
    if not task:
        return _json_error(404, "task not found")
    raw_graph = (task.graph_json or "").strip()
    if not raw_graph:
        return _json_error(400, "graph_json empty")
    try:
        payload = json.loads(raw_graph)
    except json.JSONDecodeError:
        return _json_error(400, "graph_json invalid")

    nodes, edges = _build_visual_elements(payload)
    return GraphVisualResponse(nodes=nodes, edges=edges)


@router.post("/results/graph", response_model=QueueAckResponse)
def build_graph(
    request: IDRequest, db: Session = Depends(get_db)
) -> QueueAckResponse | JSONResponse:
    if request.id <= 0:
        return _json_error(400, "id invalid")
    task = get_task_by_id(db, request.id)
    if not task:
        return _json_error(404, "task not found")
    job = get_latest_job_by_root_url(db, task.url)
    if not job or not is_crawl_job_done(job.status):
        return _json_error(400, "task not finished")

    async_result = build_graph_task.delay(request.id)
    rdb = get_redis_client()
    rdb.sadd(GRAPH_ACTIVE_SET_KEY, async_result.id)
    _cleanup_celery_active(rdb, GRAPH_ACTIVE_SET_KEY)
    pending = _queue_pending(rdb, GRAPH_QUEUE_KEY, GRAPH_ACTIVE_SET_KEY)
    return QueueAckResponse(queued=1, queue_key=GRAPH_QUEUE_KEY, pending=pending)


@router.get("/results/preprocess/status", response_model=QueueStatusResponse)
def get_preprocess_status() -> QueueStatusResponse:
    rdb = get_redis_client()
    _cleanup_celery_active(rdb, PREPROCESS_ACTIVE_SET_KEY)
    pending = _queue_pending(rdb, PREPROCESS_QUEUE_KEY, PREPROCESS_ACTIVE_SET_KEY)
    return QueueStatusResponse(pending=pending, queue_key=PREPROCESS_QUEUE_KEY)


@router.get("/results/graph/status", response_model=QueueStatusResponse)
def get_graph_status() -> QueueStatusResponse:
    rdb = get_redis_client()
    _cleanup_celery_active(rdb, GRAPH_ACTIVE_SET_KEY)
    pending = _queue_pending(rdb, GRAPH_QUEUE_KEY, GRAPH_ACTIVE_SET_KEY)
    return QueueStatusResponse(pending=pending, queue_key=GRAPH_QUEUE_KEY)
