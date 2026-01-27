from __future__ import annotations

from app.core.bootstrap import ensure_dependencies, ensure_playwright_browsers
from app.core.logger import configure_logging

ensure_dependencies()
ensure_playwright_browsers()
configure_logging()

from datetime import datetime
import json
import logging

from celery import Celery

from app.config import get_settings
from app.services.crawl_service import crawl_job
from app.services.crawler_adapter import Crawl4AIAdapter
from app.services.graph_service import build_graph_for_task
from app.db import SessionLocal
from app.models import CrawlJob, SiteTask
from app.services.queue_keys import GRAPH_ACTIVE_SET_KEY, GRAPH_TASK_MAP_KEY
from app.services.redis_client import get_redis_client

settings = get_settings()
logger = logging.getLogger(__name__)

celery_app = Celery("crawl_tasks", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)

app = celery_app


def _parse_job_params(job: CrawlJob) -> dict:
    params = job.params or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return {}
    return params if isinstance(params, dict) else {}


def _is_job_done(status: str | None) -> bool:
    if not status:
        return False
    return status.strip().lower() in {"done", "completed", "success", "succeeded", "finished"}


def _update_task_after_job(db, job: CrawlJob) -> tuple[int | None, bool]:
    params = _parse_job_params(job)
    task_id = params.get("task_id")
    auto_build_graph = bool(params.get("auto_build_graph"))

    task = None
    if task_id:
        task = db.query(SiteTask).filter_by(id=int(task_id)).first()
    if not task:
        task = db.query(SiteTask).filter_by(url=job.root_url).first()
    if not task:
        return None, auto_build_graph

    done = _is_job_done(job.status)
    task.is_crawled = done
    if done and job.finished_at:
        task.crawled_at = job.finished_at
    elif not done:
        task.crawled_at = None
    task.page_count = job.crawled_count or task.page_count
    if job.started_at and job.finished_at:
        duration_ms = int((job.finished_at - job.started_at).total_seconds() * 1000)
        task.crawl_duration_ms = max(duration_ms, 0)
    task.updated_at = datetime.utcnow()
    return int(task.id), auto_build_graph


@celery_app.task(name="crawl_job_task", bind=True)
def crawl_job_task(self, job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(CrawlJob).filter_by(job_id=job_id).first()
        if not job:
            return
        if job.status in {"RUNNING", "DONE"}:
            return
        now = datetime.utcnow()
        job.status = "RUNNING"
        job.started_at = job.started_at or now
        job.updated_at = now
        db.commit()
        db.close()

        adapter = Crawl4AIAdapter(settings)
        crawl_job(job_id, SessionLocal, settings, adapter)

        db = SessionLocal()
        job = db.query(CrawlJob).filter_by(job_id=job_id).first()
        if not job:
            return
        if job.status == "CANCELLED":
            job.finished_at = datetime.utcnow()
        elif job.status != "FAILED":
            job.status = "DONE"
            job.finished_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        task_id, auto_build_graph = _update_task_after_job(db, job)
        db.commit()
        if auto_build_graph and task_id and _is_job_done(job.status):
            async_result = build_graph_task.delay(task_id)
            rdb = get_redis_client()
            rdb.sadd(GRAPH_ACTIVE_SET_KEY, async_result.id)
            rdb.hset(GRAPH_TASK_MAP_KEY, async_result.id, int(task_id))
    except Exception as exc:  # pylint: disable=broad-except
        db.rollback()
        job = db.query(CrawlJob).filter_by(job_id=job_id).first()
        if job:
            job.status = "FAILED"
            job.error_message = str(exc)
            job.updated_at = datetime.utcnow()
            job.finished_at = job.updated_at
            _update_task_after_job(db, job)
            db.commit()
        raise
    finally:
        db.close()


@celery_app.task(name="build_graph_task", bind=True)
def build_graph_task(self, task_id: int) -> None:
    try:
        build_graph_for_task(task_id, SessionLocal, settings)
    except Exception:  # pylint: disable=broad-except
        logger.exception("graph build failed task_id=%s", task_id)
        raise
