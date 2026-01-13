from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import CrawlJob


def get_job(db: Session, job_id: str) -> CrawlJob | None:
    return db.query(CrawlJob).filter_by(job_id=job_id).first()


def add_job(db: Session, job: CrawlJob) -> None:
    db.add(job)


def get_latest_job_by_root_url(db: Session, root_url: str) -> CrawlJob | None:
    return (
        db.query(CrawlJob)
        .filter_by(root_url=root_url)
        .order_by(CrawlJob.created_at.desc())
        .first()
    )
