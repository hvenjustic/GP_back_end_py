from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import SiteTask


def get_task_by_id(db: Session, task_id: int) -> SiteTask | None:
    return db.query(SiteTask).filter_by(id=task_id).first()
