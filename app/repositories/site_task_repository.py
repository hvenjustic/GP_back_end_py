from __future__ import annotations

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import SiteTask


def get_task_by_id(db: Session, task_id: int) -> SiteTask | None:
    return db.query(SiteTask).filter_by(id=task_id).first()


def upsert_task_for_submission(db: Session, url: str, name: str | None) -> SiteTask:
    now = datetime.utcnow()
    cleaned = (name or "").strip()
    name_value = cleaned or None

    task = db.query(SiteTask).filter_by(url=url).first()
    if task:
        task.name = name_value
        task.site_name = name_value
        task.is_crawled = False
        task.crawled_at = None
        task.llm_processed_at = None
        task.page_count = 0
        task.graph_json = None
        task.updated_at = now
        return task

    task = SiteTask(
        url=url,
        name=name_value,
        site_name=name_value,
        is_crawled=False,
        on_sale=False,
        crawled_at=None,
        llm_processed_at=None,
        page_count=0,
        graph_json=None,
        created_at=now,
        updated_at=now,
    )
    db.add(task)
    return task


def list_tasks(db: Session, page: int, page_size: int) -> tuple[list[SiteTask], int]:
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    if page_size > 100:
        page_size = 100
    offset = (page - 1) * page_size

    total = db.query(SiteTask).count()
    items = (
        db.query(SiteTask)
        .order_by(SiteTask.id.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return items, total


def list_tasks_with_graph(db: Session, page: int, page_size: int) -> tuple[list[SiteTask], int]:
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    if page_size > 100:
        page_size = 100
    offset = (page - 1) * page_size

    base_query = (
        db.query(SiteTask)
        .filter(SiteTask.graph_json.isnot(None))
        .filter(func.length(func.trim(SiteTask.graph_json)) > 0)
    )
    total = base_query.count()
    items = (
        base_query.order_by(SiteTask.id.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return items, total


def list_tasks_pending_on_sale(
    db: Session, page: int, page_size: int
) -> tuple[list[SiteTask], int]:
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    if page_size > 100:
        page_size = 100
    offset = (page - 1) * page_size

    graph_ready = func.length(func.trim(SiteTask.graph_json)) > 0
    base_query = (
        db.query(SiteTask)
        .filter(SiteTask.on_sale.is_(False))
        .filter(SiteTask.graph_json.isnot(None))
        .filter(graph_ready)
    )
    total = base_query.count()
    items = (
        base_query.order_by(SiteTask.id.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return items, total


def get_tasks_by_ids(db: Session, ids: list[int]) -> list[SiteTask]:
    if not ids:
        return []
    return db.query(SiteTask).filter(SiteTask.id.in_(ids)).all()


def list_geo_locations(db: Session, limit: int) -> list[SiteTask]:
    if limit <= 0:
        limit = 1000
    if limit > 5000:
        limit = 5000
    return (
        db.query(SiteTask)
        .filter(SiteTask.geo_location.isnot(None))
        .filter(SiteTask.geo_location != "")
        .order_by(SiteTask.id.desc())
        .limit(limit)
        .all()
    )
