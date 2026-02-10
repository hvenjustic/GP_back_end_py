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
from app.services.queue_keys import (
    CRAWL_ACTIVE_SET_KEY,
    GRAPH_ACTIVE_SET_KEY,
    GRAPH_QUEUE_KEY,
    GRAPH_TASK_MAP_KEY,
)
from app.services.redis_client import get_redis_client
from app.services import embedding_service


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
        recrawl: bool = False,
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
            if recrawl:
                params_for_cleanup = build_crawl_params(settings, CrawlRequest(root_url=raw_url))
                normalized_for_cleanup = normalize_url(raw_url, raw_url, params_for_cleanup)
                roots = {raw_url}
                if normalized_for_cleanup:
                    roots.add(normalized_for_cleanup)
                old_jobs = db.query(CrawlJob.job_id).filter(CrawlJob.root_url.in_(roots)).all()
                old_job_ids = [job_id for (job_id,) in old_jobs]
                if old_job_ids:
                    db.query(SitePage).filter(SitePage.job_id.in_(old_job_ids)).delete(
                        synchronize_session=False
                    )
                    db.query(CrawlJob).filter(CrawlJob.job_id.in_(old_job_ids)).delete(
                        synchronize_session=False
                    )
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

    def build_graph(task_id: int, rebuild: bool = True) -> str:
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
            if rebuild:
                task.graph_json = None
                task.llm_processed_at = None
                task.llm_duration_ms = 0
                task.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

        async_result = build_graph_task.delay(task_id)
        rdb = get_redis_client()
        rdb.sadd(GRAPH_ACTIVE_SET_KEY, async_result.id)
        rdb.hset(GRAPH_TASK_MAP_KEY, async_result.id, int(task_id))
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
                "available": enqueue_available,
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

    def crawl_tasks_list(category: str = "available", limit: int = 20) -> str:
        cat = (category or "").strip().lower()
        limit = max(1, min(int(limit or 20), 200))
        db = SessionLocal()
        try:
            if cat in {"queued", "pending", "running", "failed"}:
                status_map = {
                    "queued": "PENDING",
                    "pending": "PENDING",
                    "running": "RUNNING",
                    "failed": "FAILED",
                }
                status = status_map.get(cat)
                jobs = (
                    db.query(CrawlJob)
                    .filter(CrawlJob.status == status)
                    .order_by(CrawlJob.updated_at.desc())
                    .limit(limit)
                    .all()
                )
                task_ids: set[int] = set()
                root_urls: set[str] = set()
                for job in jobs:
                    params = _parse_job_params(job.params)
                    task_id = _coerce_int(params.get("task_id"))
                    if task_id:
                        task_ids.add(task_id)
                    if job.root_url:
                        root_urls.add(job.root_url)

                tasks_by_id: dict[int, SiteTask] = {}
                if task_ids:
                    tasks = db.query(SiteTask).filter(SiteTask.id.in_(task_ids)).all()
                    tasks_by_id = {int(task.id): task for task in tasks}

                tasks_by_url: dict[str, SiteTask] = {}
                if root_urls:
                    tasks = db.query(SiteTask).filter(SiteTask.url.in_(root_urls)).all()
                    tasks_by_url = {task.url: task for task in tasks}

                items: list[dict[str, Any]] = []
                for job in jobs:
                    params = _parse_job_params(job.params)
                    task_id = _coerce_int(params.get("task_id"))
                    task = tasks_by_id.get(task_id) if task_id is not None else None
                    if not task and job.root_url:
                        task = tasks_by_url.get(job.root_url)
                    items.append(
                        {
                            "job_id": job.job_id,
                            "task_id": int(task.id) if task else None,
                            "name": (task.site_name or task.name) if task else None,
                            "url": task.url if task else job.root_url,
                            "status": job.status,
                            "updated_at": job.updated_at,
                            "error_message": job.error_message,
                        }
                    )
                return json.dumps(
                    {"status": "ok", "category": cat, "count": len(items), "items": items},
                    ensure_ascii=False,
                    default=str,
                )

            if cat in {"available", "ready"}:
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

                query = db.query(SiteTask)
                if active_task_ids:
                    query = query.filter(~SiteTask.id.in_(active_task_ids))
                if active_root_urls:
                    query = query.filter(~SiteTask.url.in_(active_root_urls))
                tasks = query.order_by(SiteTask.id.desc()).limit(limit).all()
                items = [
                    {
                        "id": int(task.id),
                        "name": task.name,
                        "site_name": task.site_name,
                        "url": task.url,
                        "is_crawled": bool(task.is_crawled),
                        "updated_at": task.updated_at,
                    }
                    for task in tasks
                ]
                return json.dumps(
                    {"status": "ok", "category": cat, "count": len(items), "items": items},
                    ensure_ascii=False,
                    default=str,
                )

            if cat in {"completed", "crawled"}:
                tasks = (
                    db.query(SiteTask)
                    .filter(SiteTask.is_crawled.is_(True))
                    .order_by(SiteTask.id.desc())
                    .limit(limit)
                    .all()
                )
                items = [
                    {
                        "id": int(task.id),
                        "name": task.name,
                        "site_name": task.site_name,
                        "url": task.url,
                        "crawled_at": task.crawled_at,
                        "updated_at": task.updated_at,
                    }
                    for task in tasks
                ]
                return json.dumps(
                    {"status": "ok", "category": cat, "count": len(items), "items": items},
                    ensure_ascii=False,
                    default=str,
                )

            return json.dumps(
                {"status": "error", "error": "category invalid"},
                ensure_ascii=False,
            )
        finally:
            db.close()

    def crawl_tasks_enqueue(task_ids: list[int] | None = None, recrawl: bool = False) -> str:
        ids = [int(item) for item in (task_ids or []) if int(item) > 0]
        if not ids:
            return json.dumps({"status": "error", "error": "task_ids empty"}, ensure_ascii=False)

        db = SessionLocal()
        try:
            tasks = db.query(SiteTask).filter(SiteTask.id.in_(ids)).all()
            if not tasks:
                return json.dumps({"status": "error", "error": "tasks not found"}, ensure_ascii=False)

            queued = 0
            queued_ids: list[int] = []
            rdb = get_redis_client()
            for task in tasks:
                raw_url = (task.url or "").strip()
                if not raw_url:
                    continue
                if recrawl:
                    params_for_cleanup = build_crawl_params(settings, CrawlRequest(root_url=raw_url))
                    normalized_for_cleanup = normalize_url(raw_url, raw_url, params_for_cleanup)
                    roots = {raw_url}
                    if normalized_for_cleanup:
                        roots.add(normalized_for_cleanup)
                    old_jobs = db.query(CrawlJob.job_id).filter(CrawlJob.root_url.in_(roots)).all()
                    old_job_ids = [job_id for (job_id,) in old_jobs]
                    if old_job_ids:
                        db.query(SitePage).filter(SitePage.job_id.in_(old_job_ids)).delete(
                            synchronize_session=False
                        )
                        db.query(CrawlJob).filter(CrawlJob.job_id.in_(old_job_ids)).delete(
                            synchronize_session=False
                        )

                now = datetime.utcnow()
                task.crawl_count = int(task.crawl_count or 0) + 1
                task.is_crawled = False
                task.crawled_at = None
                task.page_count = 0
                task.graph_json = None
                task.llm_processed_at = None
                task.llm_duration_ms = 0
                task.crawl_duration_ms = 0
                task.updated_at = now

                params = build_crawl_params(settings, CrawlRequest(root_url=raw_url))
                normalized_root = normalize_url(raw_url, raw_url, params)
                if not normalized_root:
                    continue

                job_id = str(uuid4())
                job_params = params.to_dict()
                job_params["task_id"] = int(task.id)
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

                crawl_job_task.delay(job_id)
                rdb.sadd(CRAWL_ACTIVE_SET_KEY, job_id)
                queued += 1
                queued_ids.append(int(task.id))

            return json.dumps(
                {"status": "queued", "queued": queued, "task_ids": queued_ids},
                ensure_ascii=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            db.rollback()
            return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)
        finally:
            db.close()
    def graph_status() -> str:
        db = SessionLocal()
        try:
            total_tasks = db.query(SiteTask).count()
            completed_count = db.query(SiteTask).filter(SiteTask.llm_processed_at.isnot(None)).count()

            graph_has_data = func.length(func.trim(SiteTask.graph_json)) > 0
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
            repeatable_count = (
                db.query(SiteTask)
                .filter(SiteTask.is_crawled.is_(True))
                .filter(SiteTask.llm_processed_at.isnot(None))
                .filter(SiteTask.graph_json.isnot(None))
                .filter(graph_has_data)
                .count()
            )
            not_buildable_count = db.query(SiteTask).filter(SiteTask.is_crawled.is_(False)).count()

            rdb = get_redis_client()
            queue_count = int(rdb.llen(GRAPH_QUEUE_KEY) + rdb.scard(GRAPH_ACTIVE_SET_KEY))

            payload = {
                "queue": {"count": queue_count},
                "completed": completed_count,
                "buildable": buildable_count,
                "repeatable": repeatable_count,
                "not_buildable": not_buildable_count,
                "total_tasks": total_tasks,
            }
            return json.dumps(payload, ensure_ascii=False, default=str)
        finally:
            db.close()

    def graph_tasks_list(category: str = "buildable", limit: int = 20) -> str:
        cat = (category or "").strip().lower()
        limit = max(1, min(int(limit or 20), 200))
        db = SessionLocal()
        try:
            graph_has_data = func.length(func.trim(SiteTask.graph_json)) > 0
            query = db.query(SiteTask)
            if cat in {"buildable", "ready"}:
                query = (
                    query.filter(SiteTask.is_crawled.is_(True))
                    .filter(SiteTask.llm_processed_at.is_(None))
                    .filter(
                        or_(
                            SiteTask.graph_json.is_(None),
                            func.length(func.trim(SiteTask.graph_json)) == 0,
                        )
                    )
                )
            elif cat in {"repeatable", "rebuild"}:
                query = (
                    query.filter(SiteTask.is_crawled.is_(True))
                    .filter(SiteTask.llm_processed_at.isnot(None))
                    .filter(SiteTask.graph_json.isnot(None))
                    .filter(graph_has_data)
                )
            elif cat in {"completed", "done"}:
                query = query.filter(SiteTask.llm_processed_at.isnot(None))
            elif cat in {"not_buildable", "blocked"}:
                query = query.filter(SiteTask.is_crawled.is_(False))
            elif cat in {"queued", "queue"}:
                rdb = get_redis_client()
                queue_ids = rdb.lrange(GRAPH_QUEUE_KEY, 0, max(0, limit - 1))
                active_ids = list(rdb.smembers(GRAPH_ACTIVE_SET_KEY))
                celery_ids = [str(item) for item in queue_ids + active_ids if item]
                if not celery_ids:
                    return json.dumps(
                        {"status": "ok", "category": "queued", "count": 0, "items": []},
                        ensure_ascii=False,
                    )
                mapped = rdb.hmget(GRAPH_TASK_MAP_KEY, celery_ids)
                task_ids = [int(item) for item in mapped if item and str(item).isdigit()]
                if not task_ids:
                    return json.dumps(
                        {
                            "status": "ok",
                            "category": "queued",
                            "count": 0,
                            "items": [],
                            "note": "queue ids found but no task mapping",
                        },
                        ensure_ascii=False,
                    )
                query = query.filter(SiteTask.id.in_(task_ids))
            else:
                return json.dumps(
                    {"status": "error", "error": "category invalid"},
                    ensure_ascii=False,
                )

            if cat in {"queued", "queue"}:
                tasks = query.all()
            else:
                tasks = query.order_by(SiteTask.id.desc()).limit(limit).all()
            items: list[dict[str, Any]] = []
            for task in tasks:
                items.append(
                    {
                        "id": int(task.id),
                        "name": task.name,
                        "site_name": task.site_name,
                        "url": task.url,
                        "is_crawled": bool(task.is_crawled),
                        "llm_processed_at": task.llm_processed_at,
                        "updated_at": task.updated_at,
                    }
                )
            return json.dumps(
                {"status": "ok", "category": cat, "count": len(items), "items": items},
                ensure_ascii=False,
                default=str,
            )
        finally:
            db.close()

    def build_graph_batch(task_ids: list[int] | None = None) -> str:
        ids = [int(item) for item in (task_ids or []) if int(item) > 0]
        if not ids:
            return json.dumps({"status": "error", "error": "task_ids empty"}, ensure_ascii=False)

        db = SessionLocal()
        try:
            tasks = db.query(SiteTask).filter(SiteTask.id.in_(ids)).all()
            if not tasks:
                return json.dumps({"status": "error", "error": "tasks not found"}, ensure_ascii=False)

            queued = 0
            queued_ids: list[int] = []
            rdb = get_redis_client()
            now = datetime.utcnow()
            for task in tasks:
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
                    continue
                task.graph_json = None
                task.llm_processed_at = None
                task.llm_duration_ms = 0
                task.updated_at = now
                async_result = build_graph_task.delay(int(task.id))
                rdb.sadd(GRAPH_ACTIVE_SET_KEY, async_result.id)
                rdb.hset(GRAPH_TASK_MAP_KEY, async_result.id, int(task.id))
                queued += 1
                queued_ids.append(int(task.id))

            if queued_ids:
                db.commit()

            return json.dumps(
                {"status": "queued", "queued": queued, "task_ids": queued_ids},
                ensure_ascii=False,
            )
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
                    "recrawl": {
                        "type": "boolean",
                        "description": "Delete old crawl data before crawling again.",
                        "default": False,
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
                    "task_id": {"type": "integer", "description": "Site task id."},
                    "rebuild": {
                        "type": "boolean",
                        "description": "Delete old graph data before building.",
                        "default": True,
                    },
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
            name="crawl_tasks_list",
            description="List crawl tasks by category.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "available|queued|running|failed|completed",
                        "default": "available",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max tasks to return (default 20).",
                        "default": 20,
                    },
                },
            },
            handler=crawl_tasks_list,
        )
    )

    registry.register(
        ToolDefinition(
            name="crawl_tasks_enqueue",
            description="Enqueue crawl tasks by id.",
            parameters={
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Site task ids.",
                    },
                    "recrawl": {
                        "type": "boolean",
                        "description": "Delete old crawl data before crawling again.",
                        "default": False,
                    },
                },
                "required": ["task_ids"],
            },
            handler=crawl_tasks_enqueue,
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
            name="graph_tasks_list",
            description="List tasks by graph build category.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "buildable|repeatable|completed|not_buildable|queued",
                        "default": "buildable",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max tasks to return (default 20).",
                        "default": 20,
                    },
                },
            },
            handler=graph_tasks_list,
        )
    )

    registry.register(
        ToolDefinition(
            name="build_graph_batch",
            description="Trigger knowledge graph extraction for multiple tasks.",
            parameters={
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Site task ids.",
                    }
                },
                "required": ["task_ids"],
            },
            handler=build_graph_batch,
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

    def build_embedding(task_ids: list[int], embedding_method: str = "graph2vec", reduction_method: str = "umap") -> str:
        """
        为指定任务构建高维向量和三维坐标
        
        Args:
            task_ids: 任务ID列表，例如 [2] 或 [2, 4, 5]
            embedding_method: 嵌入方法，支持 "graph2vec"(默认), "gnn" 或 "node2vec"
            reduction_method: 降维方法，支持 "umap" 或 "tsne"，默认 "umap"
        """
        ids = [int(item) for item in (task_ids or []) if int(item) > 0]
        if not ids:
            return json.dumps({"status": "error", "error": "task_ids 为空"}, ensure_ascii=False)
        
        db = SessionLocal()
        try:
            # 检查任务是否存在且有graph_json
            tasks = db.query(SiteTask).filter(SiteTask.id.in_(ids)).all()
            if not tasks:
                return json.dumps({"status": "error", "error": "任务不存在"}, ensure_ascii=False)
            
            # 检查哪些任务有图谱数据
            valid_ids = []
            invalid_ids = []
            for task in tasks:
                if task.graph_json:
                    valid_ids.append(int(task.id))
                else:
                    invalid_ids.append(int(task.id))
            
            if not valid_ids:
                return json.dumps({
                    "status": "error",
                    "error": "所有任务都没有图谱数据，请先构建知识图谱",
                    "invalid_ids": invalid_ids,
                }, ensure_ascii=False)
            
            # 计算嵌入
            results = embedding_service.compute_embeddings_for_tasks(
                settings=settings,
                db=db,
                task_ids=valid_ids,
                embedding_method=embedding_method,
                reduction_method=reduction_method,
                use_gpu=True,
                save_to_db=True,
                save_to_neo4j=True,
            )
            
            # 构建返回结果
            result_items = []
            total_duration = 0
            for r in results:
                result_items.append({
                    "task_id": r.site_id,
                    "name": r.site_name,
                    "node_count": r.node_count,
                    "edge_count": r.edge_count,
                    "embedding_dim": len(r.high_dim_embedding),
                    "coord_3d": r.coord_3d,
                    "duration_ms": r.duration_ms,
                })
                total_duration += r.duration_ms
            
            response = {
                "status": "completed",
                "processed": len(results),
                "total_duration_ms": total_duration,
                "embedding_method": embedding_method,
                "reduction_method": reduction_method,
                "results": result_items,
            }
            if invalid_ids:
                response["skipped_ids"] = invalid_ids
                response["skipped_reason"] = "没有图谱数据"
            
            return json.dumps(response, ensure_ascii=False, default=str)
            
        except Exception as exc:
            db.rollback()
            return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)
        finally:
            db.close()
    
    def embedding_status(task_ids: list[int] | None = None, limit: int = 20) -> str:
        """
        查询任务的嵌入状态和计时信息
        
        Args:
            task_ids: 任务ID列表，为空则查询所有有嵌入的任务
            limit: 返回结果数量限制，默认20
        """
        limit = max(1, min(int(limit or 20), 200))
        db = SessionLocal()
        try:
            ids = None
            if task_ids:
                ids = [int(item) for item in task_ids if int(item) > 0]
            
            # 获取状态信息
            status_list = embedding_service.get_embedding_status_for_tasks(db, ids)
            
            # 限制返回数量
            status_list = status_list[:limit]
            
            # 统计信息
            total_with_embedding = db.query(SiteTask).filter(SiteTask.embedding.isnot(None)).count()
            total_with_graph = db.query(SiteTask).filter(SiteTask.graph_json.isnot(None)).count()
            total_buildable = db.query(SiteTask).filter(
                SiteTask.graph_json.isnot(None),
                SiteTask.embedding.is_(None)
            ).count()
            
            return json.dumps({
                "status": "ok",
                "summary": {
                    "total_with_embedding": total_with_embedding,
                    "total_with_graph": total_with_graph,
                    "total_buildable": total_buildable,
                },
                "count": len(status_list),
                "items": status_list,
            }, ensure_ascii=False, default=str)
            
        finally:
            db.close()

    registry.register(
        ToolDefinition(
            name="build_embedding",
            description="为指定任务构建高维向量和三维坐标。使用Graph2Vec捕捉图谱的全局结构特征。使用示例：帮我构建任务2的三维向量；帮我构建任务2,4,5的高维向量。",
            parameters={
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "任务ID列表，例如 [2] 或 [2, 4, 5]",
                    },
                    "embedding_method": {
                        "type": "string",
                        "description": "嵌入方法：graph2vec（默认，捕捉全局结构）、gnn（图神经网络）或 node2vec（随机游走）",
                        "default": "graph2vec",
                    },
                    "reduction_method": {
                        "type": "string",
                        "description": "降维方法：umap（默认，推荐）或 tsne",
                        "default": "umap",
                    },
                },
                "required": ["task_ids"],
            },
            handler=build_embedding,
        )
    )
    
    registry.register(
        ToolDefinition(
            name="embedding_status",
            description="查询任务的嵌入状态，包括是否已构建高维向量、三维坐标、构建耗时等信息。",
            parameters={
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "要查询的任务ID列表，为空则查询所有有嵌入的任务",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制，默认20",
                        "default": 20,
                    },
                },
            },
            handler=embedding_status,
        )
    )

    return registry
