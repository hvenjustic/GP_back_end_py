from __future__ import annotations

from app.core.bootstrap import ensure_dependencies, ensure_playwright_browsers
from app.core.logger import configure_logging

ensure_dependencies()
ensure_playwright_browsers()
configure_logging()

from datetime import datetime
import logging

from celery import Celery

from app.config import get_settings
from app.services.crawl_service import crawl_job
from app.services.crawler_adapter import Crawl4AIAdapter
from app.services.graph_service import build_graph_for_task
from app.db import SessionLocal
from app.models import CrawlJob

settings = get_settings()
logger = logging.getLogger(__name__)

celery_app = Celery("crawl_tasks", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)

app = celery_app


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
        db.commit()
    except Exception as exc:  # pylint: disable=broad-except
        db.rollback()
        job = db.query(CrawlJob).filter_by(job_id=job_id).first()
        if job:
            job.status = "FAILED"
            job.error_message = str(exc)
            job.updated_at = datetime.utcnow()
            job.finished_at = job.updated_at
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
