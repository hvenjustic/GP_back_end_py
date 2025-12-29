from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

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


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _load_yaml_config() -> dict[str, Any]:
    config_path = Path(os.getenv("CONFIG_PATH", "config.yaml"))
    if not config_path.exists():
        alt_path = Path("config.yml")
        if alt_path.exists():
            config_path = alt_path
        else:
            return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    data = _load_yaml_config()
    crawl = data.get("crawl", {}) if isinstance(data, dict) else {}

    static_ext_raw = (
        crawl.get("static_extensions")
        if isinstance(crawl, dict) and "static_extensions" in crawl
        else os.getenv("STATIC_EXTENSIONS")
    )
    allowed_domains_raw = (
        crawl.get("allowed_domains")
        if isinstance(crawl, dict) and "allowed_domains" in crawl
        else os.getenv("ALLOWED_DOMAINS")
    )

    database_url = data.get("database_url") or os.getenv(
        "DATABASE_URL", "mysql+pymysql://user:password@localhost:3306/crawl_db"
    )
    redis_url = data.get("redis_url") or os.getenv("REDIS_URL", "redis://localhost:6379/0")

    return Settings(
        database_url=database_url,
        redis_url=redis_url,
        max_depth=_parse_int(crawl.get("max_depth") or os.getenv("MAX_DEPTH"), 3),
        max_pages=_parse_int(crawl.get("max_pages") or os.getenv("MAX_PAGES"), 5000),
        concurrency=_parse_int(crawl.get("concurrency") or os.getenv("CONCURRENCY"), 5),
        timeout=_parse_int(crawl.get("timeout") or os.getenv("TIMEOUT"), 20),
        retries=_parse_int(crawl.get("retries") or os.getenv("RETRIES"), 2),
        strip_query=_parse_bool(
            crawl.get("strip_query") or os.getenv("STRIP_QUERY"), False
        ),
        strip_tracking_params=_parse_bool(
            crawl.get("strip_tracking_params") or os.getenv("STRIP_TRACKING_PARAMS"), True
        ),
        static_extensions=_parse_list(static_ext_raw)
        if static_ext_raw is not None
        else DEFAULT_STATIC_EXTENSIONS.copy(),
        allowed_domains=_parse_list(allowed_domains_raw) if allowed_domains_raw is not None else [],
    )
