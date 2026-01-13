from __future__ import annotations

import inspect
import json
import logging
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from app.config import Settings

logger = logging.getLogger(__name__)


class LangExtractClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._module = self._load_module()
        self._client = self._init_client(self._module) if self._module else None

    def _load_module(self):
        try:
            import langextract  # type: ignore
        except ImportError:
            return None
        return langextract

    def _init_client(self, module):
        for cls_name in ("Client", "LangExtract"):
            cls = getattr(module, cls_name, None)
            if cls is None:
                continue
            try:
                params = inspect.signature(cls).parameters
            except (TypeError, ValueError):
                params = {}
            kwargs: dict[str, Any] = {}
            if "api_key" in params:
                kwargs["api_key"] = self._settings.langextract_api_key
            if "base_url" in params:
                kwargs["base_url"] = self._settings.langextract_site
            if "site" in params:
                kwargs["site"] = self._settings.langextract_site
            try:
                return cls(**kwargs)
            except Exception:
                logger.debug("langextract client init failed for %s", cls_name, exc_info=True)
                continue
        return None

    def extract(self, text: str, prompt: str) -> Any:
        if self._client is not None:
            return self._extract_with_client(text, prompt)
        return self._extract_with_http(text, prompt)

    def _extract_with_client(self, text: str, prompt: str) -> Any:
        for method_name in ("extract", "run", "call"):
            method = getattr(self._client, method_name, None)
            if method is None:
                continue
            try:
                return self._call_method(method, text, prompt)
            except Exception:
                logger.debug("langextract method failed: %s", method_name, exc_info=True)
                continue
        raise RuntimeError("langextract client has no usable method")

    def _call_method(self, method, text: str, prompt: str) -> Any:
        try:
            params = inspect.signature(method).parameters
        except (TypeError, ValueError):
            params = {}

        kwargs: dict[str, Any] = {}
        if "text" in params:
            kwargs["text"] = text
        elif "content" in params:
            kwargs["content"] = text
        elif "input" in params:
            kwargs["input"] = text

        if "prompt" in params:
            kwargs["prompt"] = prompt
        elif "instruction" in params:
            kwargs["instruction"] = prompt
        elif "template" in params:
            kwargs["template"] = prompt

        if kwargs:
            return method(**kwargs)
        return method(text, prompt)

    def _extract_with_http(self, text: str, prompt: str) -> Any:
        url = self._build_url()
        payload = {
            "text": text,
            "prompt": prompt,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self._settings.langextract_api_key:
            req.add_header("Authorization", f"Bearer {self._settings.langextract_api_key}")

        try:
            with urlrequest.urlopen(req, timeout=self._settings.langextract_timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            raise RuntimeError(f"langextract http error: {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"langextract http failed: {exc.reason}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def _build_url(self) -> str:
        endpoint = self._settings.langextract_endpoint.strip() or "/extract"
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        site = self._settings.langextract_site.strip()
        if not site:
            raise ValueError("langextract site is not configured")
        return f"{site.rstrip('/')}/{endpoint.lstrip('/')}"
