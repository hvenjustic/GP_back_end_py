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
from app.schemas import CrawlRequest
from app.services.crawl_service import build_crawl_params, normalize_url
from app.services.langextract_client import LangExtractClient
from app.services.neo4j_service import reset_site_graph, sync_graph_to_neo4j

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

    site_snapshot = {
        "id": int(task.id),
        "url": root_url,
        "name": (task.site_name or task.name or "").strip(),
    }

    job = get_latest_job_by_root_url(db, root_url)
    if not job:
        try:
            params = build_crawl_params(settings, CrawlRequest(root_url=root_url))
            normalized_root = normalize_url(root_url, root_url, params)
        except Exception:
            normalized_root = None
        if normalized_root and normalized_root != root_url:
            job = get_latest_job_by_root_url(db, normalized_root)
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

    try:
        reset_site_graph(settings, site_snapshot["id"])
    except Exception:  # pylint: disable=broad-except
        logger.exception("neo4j reset failed task_id=%s", task_id)

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

        logger.info(
            "langextract request task_id=%s parent_url=%s children_urls=%s markdown_len=%d",
            task_id,
            current_url,
            children,
            len(markdown),
        )

        try:
            response = client.extract(str(markdown), prompt)
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
    try:
        sync_graph_to_neo4j(settings, site_snapshot, graph_data)
    except Exception:  # pylint: disable=broad-except
        logger.exception("neo4j sync failed task_id=%s", task_id)


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
    for idx, extraction in enumerate(getattr(doc, "extractions", []) or []):
        ex_class = getattr(extraction, "extraction_class", None)
        attrs = getattr(extraction, "attributes", {}) or {}
        
        # 获取 extraction_text 并记录类型（用于调试）
        extraction_text = getattr(extraction, "extraction_text", None)
        
        # 调试日志：记录 extraction_text 的类型
        if not isinstance(extraction_text, (str, int, float, type(None))):
            logger.warning(
                "Unexpected extraction_text type for extraction #%d: type=%s value=%s",
                idx,
                type(extraction_text).__name__,
                extraction_text
            )
        
        extraction_text_str = _normalize_extraction_text(extraction_text, attrs, ex_class)
        
        if ex_class == "entity":
            name = _string_value(attrs.get("name")) or extraction_text_str
            
            # 提取额外字段（平铺结构）
            extra = {}
            for key in ("country", "stage", "role", "date"):
                value = attrs.get(key)
                if value and not isinstance(value, dict):  # 只接受简单类型
                    extra[key] = _string_value(value)
                elif isinstance(value, dict):
                    logger.warning(
                        "Nested dict found in entity attributes key=%s, converting to string",
                        key
                    )
                    extra[key] = json.dumps(value, ensure_ascii=False)
            
            entities.append(
                {
                    "name": name,
                    "type": _string_value(attrs.get("type")),
                    "description": _string_value(attrs.get("description")),
                    "extra": extra,
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


def _normalize_extraction_text(extraction_text: Any, attrs: dict[str, Any], ex_class: str | None) -> str:
    """
    将 extraction_text 规范化为字符串类型。
    这是一个防御性函数，用于处理 LLM 返回错误格式的情况。
    
    Args:
        extraction_text: 原始的 extraction_text 值
        attrs: extraction 的 attributes 字典
        ex_class: extraction_class 的值（"entity" 或 "relation"）
    
    Returns:
        规范化后的字符串
    """
    # 如果已经是字符串、整数或浮点数，直接转换
    if isinstance(extraction_text, (str, int, float)):
        return str(extraction_text).strip()
    
    # 如果是字典类型（错误格式），尝试提取有用信息
    if isinstance(extraction_text, dict):
        logger.warning(
            "extraction_text is dict (should be string): %s, attempting to fix",
            extraction_text
        )
        # 尝试从字典中提取文本
        for key in ("text", "value", "content", "name"):
            if key in extraction_text and extraction_text[key]:
                return str(extraction_text[key]).strip()
        # 如果找不到，转为 JSON 字符串
        try:
            return json.dumps(extraction_text, ensure_ascii=False)
        except Exception:
            pass
    
    # 如果是列表类型（错误格式）
    if isinstance(extraction_text, list):
        logger.warning(
            "extraction_text is list (should be string): %s, attempting to fix",
            extraction_text
        )
        # 取第一个非空元素
        for item in extraction_text:
            if item:
                return str(item).strip()
    
    # 如果是 None 或其他类型，根据 extraction_class 生成备用值
    if ex_class == "entity":
        name = _string_value(attrs.get("name"))
        if name:
            return name
    elif ex_class == "relation":
        source = _string_value(attrs.get("source"))
        relation_type = _string_value(attrs.get("type"))
        target = _string_value(attrs.get("target"))
        if source and target:
            return f"{source} {relation_type} {target}".strip()
    
    # 最后的备用方案
    logger.warning("extraction_text is invalid, using empty string as fallback")
    return ""


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
