from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import CrawlJob


def get_job(db: Session, job_id: str) -> CrawlJob | None:
    return db.query(CrawlJob).filter_by(job_id=job_id).first()


def add_job(db: Session, job: CrawlJob) -> None:
    db.add(job)
