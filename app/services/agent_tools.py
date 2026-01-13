from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from sqlalchemy import text

from app.config import Settings
from app.db import SessionLocal


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


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_enabled(self, enabled: set[str]) -> list[ToolDefinition]:
        return [tool for name, tool in self._tools.items() if name in enabled]

    def to_openai_tools(self, enabled: set[str]) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self.get_enabled(enabled)]

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

    return registry
