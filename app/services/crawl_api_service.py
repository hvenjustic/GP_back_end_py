from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.config import Settings
from app.models import CrawlJob, SitePage
from app.repositories.crawl_job_repository import add_job, get_job
from app.repositories.site_page_repository import add_page, list_pages_for_tree
from app.schemas import CrawlRequest, CrawlResponse, StatusResponse
from app.services.crawl_service import build_crawl_params, hash_url, normalize_url
from app.services.crawl_tasks import crawl_job_task
from app.services.service_errors import ServiceError


def _validate_root_url(root_url: str) -> None:
    parsed = urlparse(root_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ServiceError(
            status_code=400,
            message="root_url must be http/https and include a hostname",
        )


def create_crawl_job(
    request: CrawlRequest, db: Session, settings: Settings
) -> CrawlResponse:
    root_url = request.root_url.strip()
    _validate_root_url(root_url)

    try:
        params = build_crawl_params(settings, request)
    except ValueError as exc:
        raise ServiceError(status_code=400, message=str(exc)) from exc

    normalized_root = normalize_url(root_url, root_url, params)
    if not normalized_root:
        raise ServiceError(status_code=400, message="root_url cannot be normalized")

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
    db.commit()

    crawl_job_task.delay(job_id)

    return CrawlResponse(job_id=job_id, status_url=f"/status/{job_id}")


def get_status(job_id: str, db: Session) -> StatusResponse:
    job = get_job(db, job_id)
    if not job:
        raise ServiceError(status_code=404, message="job not found")

    progress = {
        "discovered": job.discovered_count,
        "queued": job.queued_count,
        "crawled": job.crawled_count,
        "failed": job.failed_count,
        "current_depth": job.current_depth,
    }
    timestamps = {
        "created_at": job.created_at,
        "started_at": job.started_at,
        "updated_at": job.updated_at,
        "finished_at": job.finished_at,
    }
    message = job.error_message if job.status == "FAILED" else None

    return StatusResponse(
        job_id=job.job_id,
        root_url=job.root_url,
        status=job.status,
        progress=progress,
        timestamps=timestamps,
        params=job.params,
        message=message,
    )


def cancel_job(job_id: str, db: Session) -> dict[str, str]:
    job = get_job(db, job_id)
    if not job:
        raise ServiceError(status_code=404, message="job not found")

    if job.status in {"DONE", "FAILED", "CANCELLED"}:
        return {"job_id": job_id, "status": job.status}

    job.status = "CANCELLED"
    job.updated_at = datetime.utcnow()
    db.commit()
    return {"job_id": job_id, "status": job.status}


def get_tree(job_id: str, db: Session) -> dict[str, Any]:
    pages = list_pages_for_tree(db, job_id)
    if not pages:
        raise ServiceError(status_code=404, message="job not found or no data")

    nodes: dict[str, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []

    for url, parent_url, depth in pages:
        nodes[url] = {"url": url, "depth": depth, "children": []}

    for url, parent_url, _depth in pages:
        node = nodes[url]
        if parent_url and parent_url in nodes:
            nodes[parent_url]["children"].append(node)
        else:
            roots.append(node)

    return {"job_id": job_id, "roots": roots}
