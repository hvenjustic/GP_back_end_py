from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import SitePage


def add_page(db: Session, page: SitePage) -> None:
    db.add(page)


def list_pages_for_tree(db: Session, job_id: str):
    return (
        db.query(SitePage.url, SitePage.parent_url, SitePage.depth)
        .filter_by(job_id=job_id)
        .all()
    )
