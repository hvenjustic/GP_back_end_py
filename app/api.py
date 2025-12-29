from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crawl_logic import build_crawl_params, normalize_url
from app.db import get_db
from app.models import CrawlJob, SitePage
from app.tasks import crawl_job_task

router = APIRouter()


class CrawlRequest(BaseModel):
    root_url: str
    max_depth: int | None = None
    max_pages: int | None = None
    concurrency: int | None = None
    timeout: int | None = None
    retries: int | None = None
    strip_query: bool | None = None
    strip_tracking_params: bool | None = None


class CrawlResponse(BaseModel):
    job_id: str
    status_url: str


class StatusResponse(BaseModel):
    job_id: str
    root_url: str
    status: str
    progress: dict[str, Any]
    timestamps: dict[str, Any]
    params: dict[str, Any]
    message: str | None = None


def _validate_root_url(root_url: str) -> None:
    parsed = urlparse(root_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=400, detail="root_url must be http/https and include a hostname"
        )


@router.post("/crawl", response_model=CrawlResponse)
def create_crawl(request: CrawlRequest, db: Session = Depends(get_db)) -> CrawlResponse:
    settings = get_settings()
    root_url = request.root_url.strip()
    _validate_root_url(root_url)

    try:
        params = build_crawl_params(settings, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    normalized_root = normalize_url(root_url, root_url, params)
    if not normalized_root:
        raise HTTPException(status_code=400, detail="root_url cannot be normalized")

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
    db.add(job)

    root_page = SitePage(
        job_id=job_id,
        url=normalized_root,
        childrens=[],
        parent_url=None,
        depth=0,
        crawled=False,
        crawl_status="PENDING",
    )
    db.add(root_page)
    db.commit()

    crawl_job_task.delay(job_id)

    return CrawlResponse(job_id=job_id, status_url=f"/status/{job_id}")


@router.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str, db: Session = Depends(get_db)) -> StatusResponse:
    job = db.query(CrawlJob).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

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


@router.post("/cancel/{job_id}")
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    job = db.query(CrawlJob).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if job.status in {"DONE", "FAILED", "CANCELLED"}:
        return {"job_id": job_id, "status": job.status}

    job.status = "CANCELLED"
    job.updated_at = datetime.utcnow()
    db.commit()
    return {"job_id": job_id, "status": job.status}


@router.get("/tree/{job_id}")
def get_tree(job_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    pages = (
        db.query(SitePage.url, SitePage.parent_url, SitePage.depth)
        .filter_by(job_id=job_id)
        .all()
    )
    if not pages:
        raise HTTPException(status_code=404, detail="job not found or no data")

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
