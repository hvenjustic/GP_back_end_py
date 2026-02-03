from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
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
    root_url = Column(String(1024), nullable=False)
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
    url = Column(String(1024), nullable=False)
    url_hash = Column(String(64), nullable=False)

    childrens = Column(JSON, nullable=False, default=list)
    parent_url = Column(String(1024))
    depth = Column(Integer, default=0, nullable=False)

    crawled = Column(Boolean, default=False, nullable=False)
    crawl_status = Column(String(20), default="PENDING", nullable=False)
    last_crawled = Column(DateTime)

    status_code = Column(Integer)
    title = Column(Text)
    canonical_url = Column(String(1024))
    content_hash = Column(String(64))
    fit_markdown = Column(LONGTEXT)
    processed_markdown = Column(LONGTEXT)
    error_message = Column(Text)

    __table_args__ = (
        PrimaryKeyConstraint("job_id", "url_hash", name="pk_site_pages"),
        Index("idx_site_pages_job_id", "job_id"),
        Index("idx_site_pages_job_url_hash", "job_id", "url_hash"),
        Index("idx_site_pages_job_depth", "job_id", "depth"),
    )


class SiteTask(Base):
    __tablename__ = "site_tasks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(100))
    site_name = Column(String(100))
    url = Column(String(255), nullable=False)
    geo_location = Column(JSON)
    crawled_at = Column(DateTime)
    is_crawled = Column(Boolean, default=False, nullable=False)
    on_sale = Column(Boolean, default=False, nullable=False)
    crawl_count = Column(Integer, default=0, nullable=False)
    page_count = Column(Integer, default=0, nullable=False)
    graph_json = Column(LONGTEXT)
    crawl_duration_ms = Column(BigInteger, default=0, nullable=False)
    llm_processed_at = Column(DateTime)
    llm_duration_ms = Column(BigInteger, default=0, nullable=False)
    # 图嵌入向量（高维）
    embedding = Column(JSON)
    # 降维后的三维坐标
    coord_3d = Column(JSON)
    embedding_updated_at = Column(DateTime)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, nullable=False)


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id = Column(String(36), primary_key=True)
    title = Column(String(255), nullable=False, default="New Chat")
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, nullable=False)


class AgentMessage(Base):
    __tablename__ = "agent_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("agent_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(LONGTEXT)
    status = Column(String(20), default="DONE", nullable=False)
    tool_name = Column(String(120))
    tool_payload = Column(JSON)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    __table_args__ = (
        Index("idx_agent_messages_session_id", "session_id"),
        Index("idx_agent_messages_role", "role"),
        Index("idx_agent_messages_created_at", "created_at"),
    )
