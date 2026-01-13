from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


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


class GraphBuildRequest(BaseModel):
    task_id: int


class GraphBuildResponse(BaseModel):
    task_id: int
    status: str
    celery_task_id: str | None = None


class AgentSessionCreateRequest(BaseModel):
    title: str | None = None


class AgentSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentMessageResponse(BaseModel):
    id: int
    role: str
    content: str | None = None
    status: str | None = None
    created_at: datetime | None = None


class AgentSessionDetailResponse(BaseModel):
    session: AgentSessionResponse
    messages: list[AgentMessageResponse]
