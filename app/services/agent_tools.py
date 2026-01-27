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
from sqlalchemy import func, or_, text

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

    def _parse_job_params(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}
        return {}

    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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

    def crawl_status(limit: int = 20) -> str:
        limit = max(0, min(int(limit or 0), 200))
        db = SessionLocal()
        try:
            pending_count = db.query(CrawlJob).filter(CrawlJob.status == "PENDING").count()
            running_count = db.query(CrawlJob).filter(CrawlJob.status == "RUNNING").count()
            queue_total = pending_count + running_count

            failed_count = db.query(CrawlJob).filter(CrawlJob.status == "FAILED").count()
            failed_jobs = (
                db.query(CrawlJob)
                .filter(CrawlJob.status == "FAILED")
                .order_by(CrawlJob.updated_at.desc())
                .limit(limit)
                .all()
            )

            active_jobs = (
                db.query(CrawlJob)
                .filter(CrawlJob.status.in_(["PENDING", "RUNNING"]))
                .all()
            )
            active_task_ids: set[int] = set()
            active_root_urls: set[str] = set()
            for job in active_jobs:
                params = _parse_job_params(job.params)
                task_id = _coerce_int(params.get("task_id"))
                if task_id:
                    active_task_ids.add(task_id)
                if job.root_url:
                    active_root_urls.add(job.root_url)

            total_tasks = db.query(SiteTask).count()
            available_query = db.query(SiteTask)
            if active_task_ids:
                available_query = available_query.filter(~SiteTask.id.in_(active_task_ids))
            if active_root_urls:
                available_query = available_query.filter(~SiteTask.url.in_(active_root_urls))
            enqueue_available = available_query.count()

            failed_task_ids: set[int] = set()
            failed_urls: set[str] = set()
            for job in failed_jobs:
                params = _parse_job_params(job.params)
                task_id = _coerce_int(params.get("task_id"))
                if task_id:
                    failed_task_ids.add(task_id)
                if job.root_url:
                    failed_urls.add(job.root_url)

            tasks_by_id: dict[int, SiteTask] = {}
            if failed_task_ids:
                tasks = db.query(SiteTask).filter(SiteTask.id.in_(failed_task_ids)).all()
                tasks_by_id = {int(task.id): task for task in tasks}

            tasks_by_url: dict[str, SiteTask] = {}
            if failed_urls:
                tasks = db.query(SiteTask).filter(SiteTask.url.in_(failed_urls)).all()
                tasks_by_url = {task.url: task for task in tasks}

            failed_sites: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for job in failed_jobs:
                params = _parse_job_params(job.params)
                task_id = _coerce_int(params.get("task_id"))
                task = tasks_by_id.get(task_id) if task_id is not None else None
                if not task and job.root_url:
                    task = tasks_by_url.get(job.root_url)

                name = ""
                url = ""
                if task:
                    name = (task.site_name or task.name or "").strip()
                    url = (task.url or "").strip()
                if not name:
                    parsed = urlparse(job.root_url or "")
                    name = parsed.netloc or ""
                url = url or (job.root_url or "")
                key = (job.job_id or "", url)
                if key in seen:
                    continue
                seen.add(key)
                failed_sites.append(
                    {
                        "job_id": job.job_id,
                        "task_id": int(task.id) if task else None,
                        "name": name,
                        "url": url,
                        "error_message": job.error_message,
                        "updated_at": job.updated_at,
                    }
                )

            payload = {
                "queue": {
                    "pending": pending_count,
                    "running": running_count,
                    "total": queue_total,
                },
                "enqueue_available": enqueue_available,
                "failed": {
                    "count": failed_count,
                    "limit": limit,
                    "sites": failed_sites,
                },
                "total_tasks": total_tasks,
            }
            return json.dumps(payload, ensure_ascii=False, default=str)
        finally:
            db.close()

    def graph_status() -> str:
        db = SessionLocal()
        try:
            total_tasks = db.query(SiteTask).count()
            completed_count = (
                db.query(SiteTask).filter(SiteTask.llm_processed_at.isnot(None)).count()
            )

            graph_has_data = func.length(func.trim(SiteTask.graph_json)) > 0
            in_queue_count = (
                db.query(SiteTask)
                .filter(SiteTask.llm_processed_at.is_(None))
                .filter(SiteTask.graph_json.isnot(None))
                .filter(graph_has_data)
                .count()
            )
            buildable_count = (
                db.query(SiteTask)
                .filter(SiteTask.is_crawled.is_(True))
                .filter(SiteTask.llm_processed_at.is_(None))
                .filter(
                    or_(
                        SiteTask.graph_json.is_(None),
                        func.length(func.trim(SiteTask.graph_json)) == 0,
                    )
                )
                .count()
            )
            not_buildable_count = (
                db.query(SiteTask).filter(SiteTask.is_crawled.is_(False)).count()
            )

            payload = {
                "queue": {"count": in_queue_count},
                "completed": completed_count,
                "buildable": buildable_count,
                "not_buildable": not_buildable_count,
                "total_tasks": total_tasks,
            }
            return json.dumps(payload, ensure_ascii=False, default=str)
        finally:
            db.close()

    def product_status() -> str:
        db = SessionLocal()
        try:
            total_tasks = db.query(SiteTask).count()
            on_sale_count = db.query(SiteTask).filter(SiteTask.on_sale.is_(True)).count()
            graph_ready = func.length(func.trim(SiteTask.graph_json)) > 0
            pending_count = (
                db.query(SiteTask)
                .filter(SiteTask.on_sale.is_(False))
                .filter(SiteTask.graph_json.isnot(None))
                .filter(graph_ready)
                .count()
            )
            payload = {
                "on_sale": on_sale_count,
                "pending": pending_count,
                "total_tasks": total_tasks,
            }
            return json.dumps(payload, ensure_ascii=False, default=str)
        finally:
            db.close()

    def set_products_on_sale(ids: list[int] | None = None, all_pending: bool = False) -> str:
        ids_list = [int(item) for item in (ids or []) if int(item) > 0]
        if not ids_list and not all_pending:
            return json.dumps(
                {"status": "error", "error": "ids empty and all_pending false"},
                ensure_ascii=False,
            )

        db = SessionLocal()
        try:
            graph_ready = (
                SiteTask.graph_json.isnot(None)
                & (func.length(func.trim(SiteTask.graph_json)) > 0)
            )
            query = (
                db.query(SiteTask)
                .filter(SiteTask.on_sale.is_(False))
                .filter(graph_ready)
            )
            if not all_pending:
                query = query.filter(SiteTask.id.in_(ids_list))
            tasks = query.all()
            if not tasks:
                return json.dumps({"status": "no_change", "updated": 0, "ids": []}, ensure_ascii=False)

            now = datetime.utcnow()
            updated_ids: list[int] = []
            for task in tasks:
                task.on_sale = True
                task.updated_at = now
                updated_ids.append(int(task.id))
            db.commit()

            return json.dumps(
                {
                    "status": "updated",
                    "mode": "all_pending" if all_pending else "ids",
                    "updated": len(updated_ids),
                    "ids": updated_ids,
                },
                ensure_ascii=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            db.rollback()
            return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)
        finally:
            db.close()

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

    registry.register(
        ToolDefinition(
            name="crawl_status",
            description="Get crawl queue progress and failed site list.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max failed sites to return (default 20).",
                        "default": 20,
                    }
                },
            },
            handler=crawl_status,
        )
    )

    registry.register(
        ToolDefinition(
            name="graph_status",
            description="Get knowledge graph build progress summary.",
            parameters={"type": "object", "properties": {}},
            handler=graph_status,
        )
    )

    registry.register(
        ToolDefinition(
            name="product_status",
            description="Get product on-sale status counts.",
            parameters={"type": "object", "properties": {}},
            handler=product_status,
        )
    )

    registry.register(
        ToolDefinition(
            name="set_products_on_sale",
            description="Set products as on-sale when graph is ready.",
            parameters={
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Product task ids to set on sale.",
                    },
                    "all_pending": {
                        "type": "boolean",
                        "description": "Set all pending products on sale.",
                        "default": False,
                    },
                },
            },
            handler=set_products_on_sale,
        )
    )

    return registry
