from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any


@dataclass
class CrawlResult:
    url: str
    final_url: str | None
    title: str | None
    status_code: int | None
    success: bool
    error_message: str | None
    links: list[str]
    internal_links: list[str] | None
    external_links: list[str] | None
    canonical_url: str | None
    html: str | None
    fit_markdown: str | None


class Crawl4AIAdapter:
    def __init__(self) -> None:
        self._module = self._load_module()
        self._crawler, self._is_async = self._init_crawler(self._module)

    def _load_module(self):
        try:
            import crawl4ai  # type: ignore
        except ImportError as exc:
            raise ImportError("crawl4ai is not installed") from exc
        return crawl4ai

    def _init_crawler(self, module):
        if hasattr(module, "AsyncWebCrawler"):
            return module.AsyncWebCrawler(), True
        if hasattr(module, "WebCrawler"):
            return module.WebCrawler(), False
        if hasattr(module, "Crawler"):
            return module.Crawler(), False
        raise RuntimeError("Unsupported crawl4ai crawler type")

    def _build_config(self, timeout: int):
        config_cls = getattr(self._module, "CrawlerRunConfig", None) or getattr(
            self._module, "RunConfig", None
        )
        if not config_cls:
            return None
        try:
            return config_cls(timeout=timeout)
        except TypeError:
            return config_cls()

    async def fetch(self, url: str, timeout: int) -> CrawlResult:
        config = self._build_config(timeout)
        if self._is_async:
            result = await self._call_async(url, config)
        else:
            result = await asyncio.to_thread(self._call_sync, url, config)
        return self._parse_result(url, result)

    async def _call_async(self, url: str, config) -> Any:
        for method_name in ("arun", "run", "crawl"):
            method = getattr(self._crawler, method_name, None)
            if not method:
                continue
            result = self._call_with_config(method, url, config)
            if inspect.isawaitable(result):
                return await result
            return result
        raise RuntimeError("crawl4ai crawler has no usable async method")

    def _call_sync(self, url: str, config) -> Any:
        for method_name in ("run", "crawl", "arun"):
            method = getattr(self._crawler, method_name, None)
            if not method:
                continue
            result = self._call_with_config(method, url, config)
            if inspect.isawaitable(result):
                return asyncio.run(result)
            return result
        raise RuntimeError("crawl4ai crawler has no usable sync method")

    def _call_with_config(self, method, url: str, config):
        if config is None:
            return method(url)
        try:
            return method(url, config=config)
        except TypeError:
            return method(url)

    def _parse_result(self, url: str, result: Any) -> CrawlResult:
        final_url = _get_attr(result, ["final_url", "url", "finalUrl"]) or url
        title = _get_attr(result, ["title"])
        status_code = _get_attr(result, ["status_code", "status", "statusCode"])
        canonical_url = _get_attr(result, ["canonical_url", "canonical"])
        html = _get_attr(result, ["html", "raw_html", "content", "markdown"])

        markdown_obj = _get_attr(result, ["markdown"])
        fit_markdown = None
        if markdown_obj is not None:
            if isinstance(markdown_obj, dict):
                fit_markdown = markdown_obj.get("fit_markdown")
            else:
                fit_markdown = getattr(markdown_obj, "fit_markdown", None)

        success = _get_attr(result, ["success"])
        error_message = _get_attr(result, ["error_message", "error"])
        if success is None:
            success = bool(html)
        if success:
            error_message = None

        links_obj = _get_attr(result, ["links", "link", "urls"])
        internal_links = _get_attr(result, ["internal_links", "internal_urls"])
        external_links = _get_attr(result, ["external_links", "external_urls"])

        if isinstance(links_obj, dict):
            internal_links = internal_links or links_obj.get("internal")
            external_links = external_links or links_obj.get("external")
            links = _extract_link_list(links_obj.get("all"))
        else:
            links = _extract_link_list(links_obj)

        internal_list = _extract_link_list(internal_links) if internal_links else None
        external_list = _extract_link_list(external_links) if external_links else None

        if not links and (internal_list or external_list):
            combined = []
            if internal_list:
                combined.extend(internal_list)
            if external_list:
                combined.extend(external_list)
            links = combined

        return CrawlResult(
            url=url,
            final_url=final_url,
            title=title,
            status_code=_coerce_int(status_code),
            success=bool(success),
            error_message=error_message,
            links=links or [],
            internal_links=internal_list,
            external_links=external_list,
            canonical_url=canonical_url,
            html=html,
            fit_markdown=fit_markdown,
        )
    async def aclose(self) -> None:
        if not self._is_async:
            return
        close_method = getattr(self._crawler, "close", None)
        if not close_method:
            return
        result = close_method()
        if inspect.isawaitable(result):
            await result



def _get_attr(obj: Any, names: list[str]) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _extract_link_list(items: Any) -> list[str]:
    if not items:
        return []
    if isinstance(items, (list, tuple, set)):
        values = []
        for item in items:
            link = _extract_single_link(item)
            if link:
                values.append(link)
        return values
    link = _extract_single_link(items)
    return [link] if link else []


def _extract_single_link(item: Any) -> str | None:
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("href") or item.get("url") or item.get("link")
    for attr in ("href", "url", "link"):
        if hasattr(item, attr):
            value = getattr(item, attr)
            if value:
                return value
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
