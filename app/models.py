from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.utcnow()


class CrawlJob(Base):
    __tablename__ = "crawl_jobs"

    job_id = Column(String(36), primary_key=True)
    root_url = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="PENDING")

    created_at = Column(DateTime, default=utcnow, nullable=False)
    started_at = Column(DateTime)
    updated_at = Column(DateTime, default=utcnow, nullable=False)
    finished_at = Column(DateTime)

    discovered_count = Column(Integer, default=0, nullable=False)
    queued_count = Column(Integer, default=0, nullable=False)
    crawled_count = Column(Integer, default=0, nullable=False)
    failed_count = Column(Integer, default=0, nullable=False)
    current_depth = Column(Integer, default=0, nullable=False)

    error_message = Column(Text)
    params = Column(JSON, nullable=False, default=dict)


class SitePage(Base):
    __tablename__ = "site_pages"

    job_id = Column(String(36), nullable=False)
    url = Column(Text, nullable=False)

    childrens = Column(JSON, nullable=False, default=list)
    parent_url = Column(Text)
    depth = Column(Integer, default=0, nullable=False)

    crawled = Column(Boolean, default=False, nullable=False)
    crawl_status = Column(String(20), default="PENDING", nullable=False)
    last_crawled = Column(DateTime)

    status_code = Column(Integer)
    title = Column(Text)
    canonical_url = Column(Text)
    content_hash = Column(String(64))
    fit_markdown = Column(LONGTEXT)
    error_message = Column(Text)

    __table_args__ = (
        PrimaryKeyConstraint("job_id", "url", name="pk_site_pages"),
        Index("idx_site_pages_job_id", "job_id"),
        Index("idx_site_pages_job_depth", "job_id", "depth"),
    )


class SiteEdge(Base):
    __tablename__ = "site_edges"

    job_id = Column(String(36), nullable=False)
    src_url = Column(Text, nullable=False)
    dst_url = Column(Text, nullable=False)
    anchor_text = Column(Text)
    is_internal = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("job_id", "src_url", "dst_url", name="pk_site_edges"),
        Index("idx_site_edges_job_id", "job_id"),
    )

