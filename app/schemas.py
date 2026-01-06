from __future__ import annotations

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
