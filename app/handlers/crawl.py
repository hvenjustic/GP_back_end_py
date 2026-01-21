from __future__ import annotations

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.schemas import CrawlRequest, CrawlResponse, StatusResponse
from app.services import crawl_api_service
from app.services.service_errors import ServiceError


def create_crawl(
    request: CrawlRequest, db: Session = Depends(get_db)
) -> CrawlResponse:
    try:
        settings = get_settings()
        return crawl_api_service.create_crawl_job(request, db, settings)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def get_status(job_id: str, db: Session = Depends(get_db)) -> StatusResponse:
    try:
        return crawl_api_service.get_status(job_id, db)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def cancel_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        return crawl_api_service.cancel_job(job_id, db)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def get_tree(job_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return crawl_api_service.get_tree(job_id, db)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
