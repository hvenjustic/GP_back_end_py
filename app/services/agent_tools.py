from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Callable
from urllib.parse import urlparse
from uuid import uuid4
from urllib.request import Request, urlopen

from langchain.tools import StructuredTool
from langchain_core.tools import BaseTool
from sqlalchemy import text

from app.config import Settings
from app.db import SessionLocal
from app.models import CrawlJob, SitePage, SiteTask
from app.repositories.crawl_job_repository import get_latest_job_by_root_url
from app.repositories.site_task_repository import upsert_task_for_submission
from app.schemas import CrawlRequest
from app.services.crawl_service import build_crawl_params, hash_url, normalize_url
from app.services.crawl_tasks import build_graph_task, crawl_job_task
from app.services.queue_keys import CRAWL_ACTIVE_SET_KEY, GRAPH_ACTIVE_SET_KEY
from app.services.redis_client import get_redis_client


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_langchain_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.handler,
            name=self.name,
            description=self.description,
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_enabled(self, enabled: set[str]) -> list[ToolDefinition]:
        return [tool for name, tool in self._tools.items() if name in enabled]

    def to_openai_tools(self, enabled: set[str]) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self.get_enabled(enabled)]

    def to_langchain_tools(self, enabled: set[str]) -> list[BaseTool]:
        return [tool.to_langchain_tool() for tool in self.get_enabled(enabled)]

    def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"tool not found: {name}")
        return tool.handler(**args)


def build_default_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()

    def http_get(url: str, timeout: int = 10) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("only http/https urls are allowed")
        if not settings.agent_http_allowlist:
            raise ValueError("http allowlist is not configured")
        hostname = (parsed.hostname or "").lower()
        allow = {host.lower() for host in settings.agent_http_allowlist}
        if hostname not in allow:
            raise ValueError("hostname not allowed")
        req = Request(url, headers={"User-Agent": "GP-Agent/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read(200_000)
        return raw.decode("utf-8", errors="replace")

    def db_query(query: str) -> str:
        normalized = " ".join(query.strip().split())
        if not normalized.lower().startswith("select "):
            raise ValueError("only SELECT statements are allowed")
        if ";" in normalized[:-1]:
            raise ValueError("multiple statements are not allowed")
        if " limit " not in normalized.lower():
            limit = max(settings.agent_sql_max_rows, 1)
            normalized = f"{normalized} LIMIT {limit}"
        db = SessionLocal()
        try:
            result = db.execute(text(normalized))
            rows = result.mappings().all()
            return json.dumps(rows, ensure_ascii=False, default=str)
        finally:
            db.close()

    def crawl_site(
        url: str,
        name: str | None = None,
        auto_build_graph: bool = True,
        max_depth: int | None = None,
        max_pages: int | None = None,
        concurrency: int | None = None,
        timeout: int | None = None,
        retries: int | None = None,
        strip_query: bool | None = None,
        strip_tracking_params: bool | None = None,
    ) -> str:
        raw_url = (url or "").strip()
        if not raw_url:
            return json.dumps({"status": "error", "error": "url empty"}, ensure_ascii=False)
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return json.dumps(
                {"status": "error", "error": "url must be http/https and include a hostname"},
                ensure_ascii=False,
            )

        cleaned_name = (name or "").strip() or parsed.netloc.strip()
        request = CrawlRequest(
            root_url=raw_url,
            max_depth=max_depth,
            max_pages=max_pages,
            concurrency=concurrency,
            timeout=timeout,
            retries=retries,
            strip_query=strip_query,
            strip_tracking_params=strip_tracking_params,
        )
        params = build_crawl_params(settings, request)
        normalized_root = normalize_url(raw_url, raw_url, params)
        if not normalized_root:
            return json.dumps(
                {"status": "error", "error": "url cannot be normalized"},
                ensure_ascii=False,
            )

        db = SessionLocal()
        try:
            now = datetime.utcnow()
            task = upsert_task_for_submission(db, raw_url, cleaned_name)
            db.flush()
            task_id = int(task.id)
            task.crawl_count = int(task.crawl_count or 0) + 1
            task.is_crawled = False
            task.crawled_at = None
            task.page_count = 0
            task.graph_json = None
            task.llm_processed_at = None
            task.llm_duration_ms = 0
            task.crawl_duration_ms = 0
            task.updated_at = now

            job_id = str(uuid4())
            job_params = params.to_dict()
            job_params["task_id"] = task_id
            if auto_build_graph:
                job_params["auto_build_graph"] = True
            job = CrawlJob(
                job_id=job_id,
                root_url=normalized_root,
                status="PENDING",
                created_at=now,
                updated_at=now,
                discovered_count=1,
                queued_count=1,
                params=job_params,
            )
            root_page = SitePage(
                job_id=job_id,
                url=normalized_root,
                url_hash=hash_url(normalized_root),
                childrens=[],
                parent_url=None,
                depth=0,
                crawled=False,
                crawl_status="PENDING",
            )
            db.add(job)
            db.add(root_page)
            db.commit()
        except Exception as exc:  # pylint: disable=broad-except
            db.rollback()
            return json.dumps(
                {"status": "error", "error": str(exc)},
                ensure_ascii=False,
            )
        finally:
            db.close()

        crawl_job_task.delay(job_id)
        rdb = get_redis_client()
        rdb.sadd(CRAWL_ACTIVE_SET_KEY, job_id)

        return json.dumps(
            {
                "status": "queued",
                "task_id": task_id,
                "job_id": job_id,
                "status_url": f"/status/{job_id}",
                "auto_build_graph": bool(auto_build_graph),
            },
            ensure_ascii=False,
        )

    def build_graph(task_id: int) -> str:
        if task_id <= 0:
            return json.dumps({"status": "error", "error": "task_id invalid"}, ensure_ascii=False)
        db = SessionLocal()
        try:
            task = db.query(SiteTask).filter_by(id=task_id).first()
            if not task:
                return json.dumps({"status": "error", "error": "task not found"}, ensure_ascii=False)
            job = get_latest_job_by_root_url(db, task.url)
            if not job:
                params = build_crawl_params(settings, CrawlRequest(root_url=task.url))
                normalized = normalize_url(task.url, task.url, params)
                if normalized and normalized != task.url:
                    job = get_latest_job_by_root_url(db, normalized)
            if not job or not job.status or job.status.strip().lower() not in {
                "done",
                "completed",
                "success",
                "succeeded",
                "finished",
            }:
                return json.dumps(
                    {"status": "not_ready", "error": "crawl job not finished"},
                    ensure_ascii=False,
                )
        finally:
            db.close()

        async_result = build_graph_task.delay(task_id)
        rdb = get_redis_client()
        rdb.sadd(GRAPH_ACTIVE_SET_KEY, async_result.id)
        return json.dumps(
            {"status": "queued", "task_id": task_id, "celery_task_id": async_result.id},
            ensure_ascii=False,
        )

    registry.register(
        ToolDefinition(
            name="http_get",
            description="Fetch a URL over HTTP(S). Returns response text.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout seconds (default 10).",
                        "default": 10,
                    },
                },
                "required": ["url"],
            },
            handler=http_get,
        )
    )

    registry.register(
        ToolDefinition(
            name="db_query",
            description="Run a read-only SQL SELECT against the primary database.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT statement."}
                },
                "required": ["query"],
            },
            handler=db_query,
        )
    )

    registry.register(
        ToolDefinition(
            name="crawl_site",
            description="Create a crawl job for a website and optionally auto-build the graph.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Website URL to crawl."},
                    "name": {"type": "string", "description": "Optional site name."},
                    "auto_build_graph": {
                        "type": "boolean",
                        "description": "Auto build knowledge graph after crawl completes.",
                        "default": True,
                    },
                    "max_depth": {"type": "integer", "description": "Override crawl max depth."},
                    "max_pages": {"type": "integer", "description": "Override crawl max pages."},
                    "concurrency": {"type": "integer", "description": "Override crawl concurrency."},
                    "timeout": {"type": "integer", "description": "Override crawl timeout (seconds)."},
                    "retries": {"type": "integer", "description": "Override crawl retries."},
                    "strip_query": {"type": "boolean", "description": "Strip query parameters."},
                    "strip_tracking_params": {
                        "type": "boolean",
                        "description": "Strip tracking parameters.",
                    },
                },
                "required": ["url"],
            },
            handler=crawl_site,
        )
    )

    registry.register(
        ToolDefinition(
            name="build_graph",
            description="Trigger knowledge graph extraction for a task when crawl is finished.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "Site task id."}
                },
                "required": ["task_id"],
            },
            handler=build_graph,
        )
    )

    return registry
