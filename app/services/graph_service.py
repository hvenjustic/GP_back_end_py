from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from app.config import Settings
from app.models import SitePage, SiteTask
from app.repositories.crawl_job_repository import get_latest_job_by_root_url
from app.services.langextract_client import LangExtractClient

logger = logging.getLogger(__name__)
MAX_LOG_CHARS = 4000

DEFAULT_PROMPT = (
    "你是信息抽取系统。你的任务是从 Markdown 文本中抽取可用于构建知识图谱的结构化信息。"
)


def build_graph_for_task(task_id: int, session_factory, settings: Settings) -> None:
    db = session_factory()
    task = db.query(SiteTask).filter_by(id=task_id).first()
    if not task:
        db.close()
        raise ValueError(f"site_task not found: {task_id}")

    root_url = (task.url or "").strip()
    if not root_url:
        db.close()
        raise ValueError(f"site_task url empty: {task_id}")

    job = get_latest_job_by_root_url(db, root_url)
    if not job:
        db.close()
        raise ValueError(f"crawl_job not found for url: {root_url}")
    if not is_crawl_job_done(job.status):
        db.close()
        raise ValueError(f"crawl_job not finished: {job.status}")

    pages = db.query(SitePage).filter_by(job_id=job.job_id).all()
    db.close()
    if not pages:
        raise ValueError(f"site_pages empty for job: {job.job_id}")

    pages_by_url = {page.url: page for page in pages if page.url}
    start_url = job.root_url if job.root_url in pages_by_url else pick_root_url(pages)
    if not start_url:
        raise ValueError("root_url not found in site_pages")

    graph_data = _load_graph_json(task.graph_json)
    prompt = _load_prompt(settings)
    client = LangExtractClient(settings)

    start = time.monotonic()
    queue = [start_url]
    visited: set[str] = set()

    while queue:
        current_url = queue.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        page = pages_by_url.get(current_url)
        if not page:
            continue

        children = _parse_children(page.childrens)
        for child_url in children:
            if child_url in pages_by_url and child_url not in visited:
                queue.append(child_url)

        markdown = build_graph_markdown(page, children, pages_by_url)
        if not markdown:
            continue

        try:
            response = client.extract(markdown, prompt)
        except Exception as exc:
            logger.warning(
                "langextract failed task_id=%s url=%s error=%s",
                task_id,
                current_url,
                exc,
            )
            continue

        entities, relations = _extract_graph_items(response)
        if entities:
            graph_data["entities"].extend(entities)
        if relations:
            graph_data["relations"].extend(relations)

        payload = {"entities": entities, "relations": relations}
        logger.info(
            "langextract response task_id=%s url=%s entities=%s relations=%s payload=%s",
            task_id,
            current_url,
            len(entities),
            len(relations),
            _truncate_json(payload, MAX_LOG_CHARS),
        )
        _persist_graph_progress(session_factory, task_id, graph_data)

    duration_ms = int((time.monotonic() - start) * 1000)
    _finalize_graph(session_factory, task_id, graph_data, duration_ms)


def is_crawl_job_done(status: str | None) -> bool:
    if not status:
        return False
    return status.strip().lower() in {"done", "completed", "success", "succeeded", "finished"}


def pick_root_url(pages: Iterable[SitePage]) -> str:
    for page in pages:
        url = (page.url or "").strip()
        parent = (page.parent_url or "").strip()
        if url and not parent:
            return url
    first = next(iter(pages), None)
    if not first:
        return ""
    return (first.url or "").strip()


def build_graph_markdown(
    parent: SitePage, children: list[str], pages_by_url: dict[str, SitePage]
) -> str:
    sections: list[str] = []
    parent_text = _page_markdown(parent)
    if parent_text:
        sections.append(f"# parent\n{parent_text}")
    else:
        sections.append("# parent")

    has_child = False
    for idx, child_url in enumerate(children):
        child = pages_by_url.get(child_url)
        if not child:
            continue
        child_text = _page_markdown(child)
        if not child_text:
            continue
        sections.append(f"# child:{idx}\n{child_text}")
        has_child = True

    if not has_child and not parent_text:
        return ""
    return "\n\n".join(sections).strip()


def _page_markdown(page: SitePage) -> str:
    if page.fit_markdown:
        return str(page.fit_markdown).strip()
    if page.processed_markdown:
        return str(page.processed_markdown).strip()
    return ""


def _parse_children(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        raw_str = raw.strip()
        if not raw_str:
            return []
        try:
            data = json.loads(raw_str)
        except json.JSONDecodeError:
            return [raw_str]
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    return []


def _load_graph_json(raw: str | None) -> dict[str, list[Any]]:
    if not raw or not raw.strip():
        return {"entities": [], "relations": []}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}
    if isinstance(data, dict):
        entities = data.get("entities")
        relations = data.get("relations")
        return {
            "entities": entities if isinstance(entities, list) else [],
            "relations": relations if isinstance(relations, list) else [],
        }
    return {"entities": [], "relations": []}


def _extract_graph_items(doc: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for extraction in getattr(doc, "extractions", []) or []:
        ex_class = getattr(extraction, "extraction_class", None)
        attrs = getattr(extraction, "attributes", {}) or {}
        if ex_class == "entity":
            name = _string_value(attrs.get("name")) or _string_value(
                getattr(extraction, "extraction_text", None)
            )
            entities.append(
                {
                    "name": name,
                    "type": _string_value(attrs.get("type")),
                    "description": _string_value(attrs.get("description")),
                    "extra": attrs.get("extra")
                    if isinstance(attrs.get("extra"), dict)
                    else {},
                }
            )
        elif ex_class == "relation":
            relations.append(
                {
                    "source": _string_value(attrs.get("source")),
                    "target": _string_value(attrs.get("target")),
                    "type": _string_value(attrs.get("type")),
                }
            )
    return entities, relations


def _truncate_json(payload: Any, limit: int) -> str:
    try:
        raw = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        raw = str(payload)
    if len(raw) <= limit:
        return raw
    return f"{raw[:limit]}...(truncated)"


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _persist_graph_progress(session_factory, task_id: int, items: dict[str, list[Any]]) -> None:
    db = session_factory()
    task = db.query(SiteTask).filter_by(id=task_id).first()
    if not task:
        db.close()
        raise ValueError(f"site_task not found: {task_id}")
    task.graph_json = json.dumps(items, ensure_ascii=False, default=str)
    task.updated_at = datetime.utcnow()
    db.commit()
    db.close()


def _finalize_graph(
    session_factory, task_id: int, items: dict[str, list[Any]], duration_ms: int
) -> None:
    db = session_factory()
    task = db.query(SiteTask).filter_by(id=task_id).first()
    if not task:
        db.close()
        raise ValueError(f"site_task not found: {task_id}")
    task.graph_json = json.dumps(items, ensure_ascii=False, default=str)
    task.llm_processed_at = datetime.utcnow()
    task.llm_duration_ms = duration_ms
    task.updated_at = datetime.utcnow()
    db.commit()
    db.close()


def _load_prompt(settings: Settings) -> str:
    if settings.langextract_prompt_inline:
        return settings.langextract_prompt_inline.strip()
    if settings.langextract_prompt_path:
        path = Path(settings.langextract_prompt_path)
        if not path.is_absolute():
            root = Path(__file__).resolve().parent.parent.parent
            path = root / settings.langextract_prompt_path
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning("langextract prompt not found: %s", path)
    return DEFAULT_PROMPT
