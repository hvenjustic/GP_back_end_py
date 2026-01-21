from __future__ import annotations

import logging
import json
from typing import Any

import langextract as lx

from app.config import Settings

logger = logging.getLogger(__name__)
MAX_DEBUG_CHARS = 2000

EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "## About\n"
            "Acme Bio Inc. partners with Example University on a fermentation process. "
            "Acme Bio Inc. supplies ReagentX reagent for research. "
            "Dr. Jane Doe is a Scientist at Example University. "
            "Example University publishes the research paper Fermentation Breakthrough.\n"
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="Acme Bio Inc.",
                attributes={
                    "name": "Acme Bio Inc.",
                    "type": "Company",
                    "description": "A biotech company supplying research reagents.",
                    "extra": {"country": "USA"},
                },
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="Example University",
                attributes={
                    "name": "Example University",
                    "type": "University",
                    "description": "A university conducting biotechnology research.",
                    "extra": {"country": "USA"},
                },
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="fermentation process",
                attributes={
                    "name": "fermentation process",
                    "type": "Fermentation Process",
                    "description": "A fermentation process used in biotech research.",
                    "extra": {},
                },
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="ReagentX",
                attributes={
                    "name": "ReagentX",
                    "type": "Reagent",
                    "description": "A reagent supplied for laboratory research.",
                    "extra": {},
                },
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="Dr. Jane Doe",
                attributes={
                    "name": "Jane Doe",
                    "type": "Scientist",
                    "description": "A scientist at Example University.",
                    "extra": {"role": "Scientist"},
                },
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="Fermentation Breakthrough",
                attributes={
                    "name": "Fermentation Breakthrough",
                    "type": "Research Paper",
                    "description": "A research paper published by Example University.",
                    "extra": {},
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="partners with Example University",
                attributes={
                    "source": "Acme Bio Inc.",
                    "target": "Example University",
                    "type": "PARTNERS_WITH",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="supplies ReagentX reagent",
                attributes={
                    "source": "Acme Bio Inc.",
                    "target": "ReagentX",
                    "type": "SUPPLIES",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="is a Scientist at Example University",
                attributes={
                    "source": "Jane Doe",
                    "target": "Example University",
                    "type": "WORKS_AT",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="publishes the research paper Fermentation Breakthrough",
                attributes={
                    "source": "Example University",
                    "target": "Fermentation Breakthrough",
                    "type": "PUBLISHES",
                },
            ),
        ],
    )
]

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "gpt4", "gpt-4")


def _build_extract_kwargs(settings: Settings, prompt: str) -> dict[str, Any]:
    model_id = settings.langextract_model_id or "gemini-2.5-flash"
    common = dict(
        prompt_description=prompt,
        examples=EXAMPLES,
        model_id=model_id,
        extraction_passes=settings.langextract_extraction_passes,
        max_workers=settings.langextract_max_workers,
        max_char_buffer=settings.langextract_max_char_buffer,
    )

    if model_id.lower().startswith(_OPENAI_PREFIXES):
        api_key = settings.langextract_openai_api_key or settings.langextract_api_key
        if not api_key:
            logger.warning("openai api key is empty for model_id=%s", model_id)
        common.update(
            api_key=api_key,
            fence_output=True,
            use_schema_constraints=True,
            validate_json=True,  # 启用 JSON 验证
        )
        base_url = settings.langextract_openai_base_url.strip()
        if base_url:
            common["language_model_params"] = {"base_url": base_url}
        # 为 OpenAI 模型添加响应格式约束
        if "language_model_params" not in common:
            common["language_model_params"] = {}
        common["language_model_params"]["response_format"] = {"type": "json_object"}
        return common

    common.update(api_key=settings.langextract_api_key)
    return common


class LangExtractClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def extract(self, text: str, prompt: str) -> Any:
        kwargs = _build_extract_kwargs(self._settings, prompt)
        try:
            return lx.extract(text_or_documents=text, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            debug_payload = _extract_debug_payload(exc)
            if debug_payload:
                logger.warning("langextract raw response: %s", debug_payload)
            raise


def _extract_debug_payload(exc: Exception) -> str:
    candidates: list[tuple[str, Any]] = []
    for name in (
        "response",
        "raw_response",
        "raw_output",
        "content",
        "body",
        "text",
        "data",
        "payload",
    ):
        value = getattr(exc, name, None)
        if value is not None:
            candidates.append((name, value))

    extra = getattr(exc, "__dict__", None)
    if isinstance(extra, dict):
        for key, value in extra.items():
            if key in {name for name, _ in candidates}:
                continue
            if value is not None:
                candidates.append((key, value))

    if not candidates:
        for arg in getattr(exc, "args", []) or []:
            if arg:
                candidates.append(("args", arg))

    for name, value in candidates:
        rendered = _render_debug_value(value)
        if rendered:
            return f"{name}={rendered}"
    return ""


def _render_debug_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        raw = json.dumps(value, ensure_ascii=False, default=str)
    else:
        raw = str(value)
    raw = raw.strip()
    if not raw:
        return ""
    if len(raw) <= MAX_DEBUG_CHARS:
        return raw
    return f"{raw[:MAX_DEBUG_CHARS]}...(truncated)"
