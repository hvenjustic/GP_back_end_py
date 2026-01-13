from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_STATIC_EXTENSIONS = [
    ".jpg",
    ".png",
    ".gif",
    ".svg",
    ".css",
    ".js",
    ".ico",
    ".mp4",
    ".zip",
    ".rar",
    ".pdf",
]


@dataclass
class Settings:
    database_url: str
    redis_url: str
    max_depth: int = 3
    max_pages: int = 5000
    concurrency: int = 5
    timeout: int = 20
    retries: int = 2
    strip_query: bool = False
    strip_tracking_params: bool = True
    static_extensions: list[str] = field(default_factory=lambda: DEFAULT_STATIC_EXTENSIONS.copy())
    allowed_domains: list[str] = field(default_factory=list)
    markdown_filter_threshold: float = 0.22
    markdown_filter_threshold_type: str = "dynamic"
    markdown_filter_min_word_threshold: int = 0
    markdown_ignore_links: bool = True
    markdown_ignore_images: bool = True
    auto_install_deps: bool = True
    auto_install_playwright: bool = True
    worker_concurrency: int = 1
    worker_log_file: str = "worker.log"
    langextract_model_id: str = "gemini-2.5-flash"
    langextract_api_key: str = ""
    langextract_openai_api_key: str = ""
    langextract_openai_base_url: str = ""
    langextract_extraction_passes: int = 2
    langextract_max_workers: int = 12
    langextract_max_char_buffer: int = 1200
    langextract_prompt_path: str = ""
    langextract_prompt_inline: str = ""


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default



def _parse_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default



def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _parse_first_address(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            item_str = str(item).strip()
            if item_str:
                return item_str
        return None
    item_str = str(value).strip()
    return item_str or None



def _load_yaml_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_mysql_url(mysql_cfg: Any, fallback: str) -> str:
    if not isinstance(mysql_cfg, dict):
        return fallback
    addr = _parse_first_address(mysql_cfg.get("dbMysqlAddress"))
    user = mysql_cfg.get("dbMysqlUserName")
    password = mysql_cfg.get("dbMysqlPassword")
    dbname = mysql_cfg.get("dbMysqlDatabaseName")
    if addr and user and dbname and password is not None:
        return f"mysql+pymysql://{user}:{password}@{addr}/{dbname}"
    return fallback


def _build_redis_url(redis_cfg: Any, fallback: str) -> str:
    if not isinstance(redis_cfg, dict):
        return fallback
    addr = _parse_first_address(redis_cfg.get("dbAddress"))
    if not addr:
        return fallback
    db_index = _parse_int(redis_cfg.get("dbRedisDb"), 0)
    password = redis_cfg.get("dbPassWord")
    if password:
        return f"redis://:{password}@{addr}/{db_index}"
    return f"redis://{addr}/{db_index}"




@lru_cache
def get_settings() -> Settings:
    data = _load_yaml_config()
    crawl = data.get("crawl", {}) if isinstance(data, dict) else {}
    runtime = data.get("runtime", {}) if isinstance(data, dict) else {}
    langextract = data.get("langextract", {}) if isinstance(data, dict) else {}

    if not isinstance(langextract, dict):
        langextract = {}

    static_ext_raw = (
        crawl.get("static_extensions") if isinstance(crawl, dict) else None
    )
    allowed_domains_raw = (
        crawl.get("allowed_domains") if isinstance(crawl, dict) else None
    )

    markdown_cfg = crawl.get("markdown_filter", {}) if isinstance(crawl, dict) else {}
    if not isinstance(markdown_cfg, dict):
        markdown_cfg = {}

    database_url = "mysql+pymysql://user:password@localhost:3306/crawl_db"
    redis_url = "redis://localhost:6379/0"

    if isinstance(data, dict):
        if data.get("database_url"):
            database_url = data["database_url"]
        else:
            database_url = _build_mysql_url(data.get("mysql"), database_url)

        if data.get("redis_url"):
            redis_url = data["redis_url"]
        else:
            redis_url = _build_redis_url(data.get("redis"), redis_url)

    langextract_model_id = str(langextract.get("model_id") or "gemini-2.5-flash").strip()
    if not langextract_model_id:
        langextract_model_id = "gemini-2.5-flash"
    langextract_api_key = str(
        langextract.get("api_key") or langextract.get("langextract_api_key") or ""
    ).strip()
    langextract_openai_api_key = str(langextract.get("openai_api_key") or "").strip()
    langextract_openai_base_url = str(langextract.get("openai_base_url") or "").strip()
    langextract_extraction_passes = _parse_int(langextract.get("extraction_passes"), 2)
    langextract_max_workers = _parse_int(langextract.get("max_workers"), 12)
    langextract_max_char_buffer = _parse_int(langextract.get("max_char_buffer"), 1200)
    langextract_prompt_path = str(langextract.get("prompt_path") or "").strip()
    langextract_prompt_inline = str(langextract.get("prompt_inline") or "").strip()

    return Settings(
        database_url=database_url,
        redis_url=redis_url,
        max_depth=_parse_int(crawl.get("max_depth"), 3),
        max_pages=_parse_int(crawl.get("max_pages"), 5000),
        concurrency=_parse_int(crawl.get("concurrency"), 5),
        timeout=_parse_int(crawl.get("timeout"), 20),
        retries=_parse_int(crawl.get("retries"), 2),
        strip_query=_parse_bool(crawl.get("strip_query"), False),
        strip_tracking_params=_parse_bool(crawl.get("strip_tracking_params"), True),
        static_extensions=_parse_list(static_ext_raw)
        if static_ext_raw is not None
        else DEFAULT_STATIC_EXTENSIONS.copy(),
        allowed_domains=_parse_list(allowed_domains_raw) if allowed_domains_raw is not None else [],
        markdown_filter_threshold=_parse_float(
            markdown_cfg.get("threshold"),
            0.22,
        ),
        markdown_filter_threshold_type=(
            str(markdown_cfg.get("threshold_type") or "dynamic").strip() or "dynamic"
        ),
        markdown_filter_min_word_threshold=_parse_int(
            markdown_cfg.get("min_word_threshold"),
            0,
        ),
        markdown_ignore_links=_parse_bool(
            markdown_cfg.get("ignore_links"),
            True,
        ),
        markdown_ignore_images=_parse_bool(
            markdown_cfg.get("ignore_images"),
            True,
        ),
        auto_install_deps=_parse_bool(
            runtime.get("auto_install_deps") if isinstance(runtime, dict) else None, True
        ),
        auto_install_playwright=_parse_bool(
            runtime.get("auto_install_playwright") if isinstance(runtime, dict) else None, True
        ),
        worker_concurrency=_parse_int(
            runtime.get("worker_concurrency") if isinstance(runtime, dict) else None, 1
        ),
        worker_log_file=(
            str(runtime.get("worker_log_file")).strip()
            if isinstance(runtime, dict) and runtime.get("worker_log_file")
            else "worker.log"
        ),
        langextract_model_id=langextract_model_id,
        langextract_api_key=langextract_api_key,
        langextract_openai_api_key=langextract_openai_api_key,
        langextract_openai_base_url=langextract_openai_base_url,
        langextract_extraction_passes=langextract_extraction_passes,
        langextract_max_workers=langextract_max_workers,
        langextract_max_char_buffer=langextract_max_char_buffer,
        langextract_prompt_path=langextract_prompt_path,
        langextract_prompt_inline=langextract_prompt_inline,
    )
