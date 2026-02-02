from __future__ import annotations

from datetime import datetime
import inspect
import json
import logging
from typing import Any, Iterator
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import SessionLocal
from app.models import AgentMessage, AgentSession
from app.services.agent_tools import ToolRegistry, build_default_registry

logger = logging.getLogger(__name__)
MAX_TRACE_TEXT_CHARS = 600


def _patch_langchain_usage_metadata() -> None:
    try:
        from langchain_core.messages import ai as ai_messages
    except Exception:
        return
    if getattr(ai_messages, "_patched_usage_metadata", False):
        return

    original = ai_messages.add_ai_message_chunks

    def _normalize(meta: Any) -> None:
        if not meta:
            return
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if meta.get(key) is None:
                meta[key] = 0

    def safe_add(left: Any, *others: Any):
        _normalize(getattr(left, "usage_metadata", None))
        for other in others:
            _normalize(getattr(other, "usage_metadata", None))
        return original(left, *others)

    ai_messages.add_ai_message_chunks = safe_add
    ai_messages._patched_usage_metadata = True


_patch_langchain_usage_metadata()


class AgentService:
    def __init__(self, settings: Settings, registry: ToolRegistry | None = None) -> None:
        self.settings = settings
        self._llm: ChatOpenAI | None = None
        self._stream_llm: ChatOpenAI | None = None
        self._registry = registry or build_default_registry(settings)

    def stream_chat(self, message: str, session_id: str | None = None) -> Iterator[str]:
        def generator() -> Iterator[str]:
            db = SessionLocal()
            assistant_message: AgentMessage | None = None
            session: AgentSession | None = None
            trace_items: list[dict[str, Any]] = []
            trace_cursor = 0

            def _drain_trace_events() -> list[str]:
                nonlocal trace_cursor
                payloads: list[str] = []
                while trace_cursor < len(trace_items):
                    item = dict(trace_items[trace_cursor])
                    if assistant_message:
                        item["messageId"] = str(assistant_message.id)
                    if session:
                        item["sessionId"] = session.id
                    payloads.append(_sse_event("trace", item))
                    trace_cursor += 1
                return payloads

            try:
                session = self._get_or_create_session(db, session_id, message)
                self._add_message(db, session.id, "user", message)
                db.commit()

                messages = self._build_messages(db, session.id)

                assistant_message = self._add_message(
                    db, session.id, "assistant", "", status="STREAMING"
                )
                db.commit()

                yield _sse_event(
                    "meta",
                    {"label": "session_id", "value": session.id},
                )
                trace_items.append(
                    _build_trace_item(
                        step="已接收用户消息，开始准备上下文",
                        stage="prepare",
                    )
                )
                for event in _drain_trace_events():
                    yield event

                content_parts: list[str] = []

                prepared_messages = self._prepare_messages_for_stream(
                    messages,
                    traces=trace_items,
                )
                for event in _drain_trace_events():
                    yield event

                trace_items.append(
                    _build_trace_item(
                        step="上下文准备完成，开始生成最终回复",
                        stage="stream",
                    )
                )
                for event in _drain_trace_events():
                    yield event

                stream = self._get_llm(streaming=True).stream(prepared_messages)

                for chunk in stream:
                    delta = _chunk_to_text(chunk)
                    if not delta:
                        continue
                    content_parts.append(delta)
                    yield _sse_event(
                        "token",
                        {
                            "delta": delta,
                            "messageId": str(assistant_message.id),
                            "sessionId": session.id,
                        },
                    )

                final_text = "".join(content_parts).strip()
                trace_items.append(
                    _build_trace_item(
                        step="回复生成完成",
                        stage="done",
                    )
                )
                for event in _drain_trace_events():
                    yield event

                assistant_message.content = final_text
                assistant_message.status = "DONE"
                assistant_message.tool_name = "agent_trace"
                assistant_message.tool_payload = {"trace": trace_items}
                session.updated_at = datetime.utcnow()
                db.commit()

                yield _sse_event(
                    "done",
                    {
                        "messageId": str(assistant_message.id),
                        "sessionId": session.id,
                        "trace": trace_items,
                    },
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("agent stream failed")
                error_text = f"Agent error: {exc}"
                trace_items.append(
                    _build_trace_item(
                        step=f"发生错误：{exc}",
                        stage="error",
                        level="error",
                    )
                )
                if assistant_message:
                    assistant_message.content = error_text
                    assistant_message.status = "FAILED"
                    assistant_message.tool_name = "agent_trace"
                    assistant_message.tool_payload = {"trace": trace_items}
                    db.commit()
                    for event in _drain_trace_events():
                        yield event
                    yield _sse_event(
                        "token",
                        {
                            "delta": error_text,
                            "messageId": str(assistant_message.id),
                            "sessionId": session.id if session else "",
                        },
                    )
                    yield _sse_event(
                        "done",
                        {
                            "messageId": str(assistant_message.id),
                            "sessionId": session.id if session else "",
                            "trace": trace_items,
                        },
                    )
                else:
                    yield _sse_event("error", {"message": error_text})
            finally:
                db.close()

        return generator()

    def _get_llm(self, streaming: bool = False) -> ChatOpenAI:
        if streaming:
            if self._stream_llm is None:
                self._stream_llm = _build_chat_llm(self.settings, streaming=True)
            return self._stream_llm
        if self._llm is None:
            self._llm = _build_chat_llm(self.settings, streaming=False)
        return self._llm

    def _get_or_create_session(
        self, db: Session, session_id: str | None, message: str
    ) -> AgentSession:
        session = None
        if session_id:
            session = db.query(AgentSession).filter_by(id=session_id).first()

        if not session:
            session = AgentSession(id=str(uuid4()), title="New Chat")
            db.add(session)
            db.commit()

        if session.title == "New Chat":
            session.title = _truncate_title(message)
        session.updated_at = datetime.utcnow()
        db.commit()
        return session

    def _add_message(
        self,
        db: Session,
        session_id: str,
        role: str,
        content: str,
        status: str = "DONE",
    ) -> AgentMessage:
        message = AgentMessage(
            session_id=session_id,
            role=role,
            content=content,
            status=status,
        )
        db.add(message)
        return message

    def _build_messages(self, db: Session, session_id: str) -> list[BaseMessage]:
        query = (
            db.query(AgentMessage)
            .filter_by(session_id=session_id)
            .order_by(AgentMessage.id.desc())
            .limit(self.settings.agent_max_history)
        )
        items = list(reversed(query.all()))

        messages: list[BaseMessage] = []
        if self.settings.agent_system_prompt:
            messages.append(SystemMessage(content=self.settings.agent_system_prompt))

        for item in items:
            content = item.content or ""
            if item.role == "user":
                messages.append(HumanMessage(content=content))
            elif item.role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def _prepare_messages_for_stream(
        self,
        messages: list[BaseMessage],
        traces: list[dict[str, Any]] | None = None,
    ) -> list[BaseMessage]:
        trace_records = traces if isinstance(traces, list) else []
        enabled = {name.strip() for name in self.settings.agent_tools_enabled if name.strip()}
        if not (self.settings.agent_use_tools and enabled):
            trace_records.append(
                _build_trace_item(
                    step="当前为纯对话模式（未启用工具调用）",
                    stage="plan",
                )
            )
            return messages

        tools = self._registry.to_langchain_tools(enabled)
        if not tools:
            trace_records.append(
                _build_trace_item(
                    step="配置了工具模式，但未找到可用工具，改为直接回答",
                    stage="plan",
                    level="warning",
                )
            )
            return messages

        trace_records.append(
            _build_trace_item(
                step=f"已启用工具模式，可调用 {len(tools)} 个工具",
                stage="plan",
                payload={"tools": sorted(enabled)},
            )
        )

        llm = self._get_llm(streaming=False)
        llm_with_tools = llm.bind_tools(tools)
        tool_rounds = 0
        working = list(messages)

        while tool_rounds < self.settings.agent_tool_max_rounds:
            response = llm_with_tools.invoke(working)
            if not isinstance(response, AIMessage):
                trace_records.append(
                    _build_trace_item(
                        step="模型返回了非 AIMessage，停止工具规划",
                        stage="plan",
                        level="warning",
                    )
                )
                break

            planner_text = _preview_text(_chunk_to_text(response), MAX_TRACE_TEXT_CHARS)
            if planner_text:
                trace_records.append(
                    _build_trace_item(
                        step=planner_text,
                        stage="plan",
                    )
                )

            tool_calls = _extract_tool_calls(response)
            if not tool_calls:
                trace_records.append(
                    _build_trace_item(
                        step="未触发工具调用，进入最终回答阶段",
                        stage="plan",
                    )
                )
                break

            working.append(response)
            trace_records.append(
                _build_trace_item(
                    step=f"第 {tool_rounds + 1} 轮触发 {len(tool_calls)} 个工具调用",
                    stage="tool",
                )
            )
            for idx, call in enumerate(tool_calls, start=1):
                tool_name = str(call.get("name") or "").strip() or "unknown_tool"
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                trace_records.append(
                    _build_trace_item(
                        step=f"执行工具：{tool_name}",
                        stage="tool",
                        payload={
                            "round": tool_rounds + 1,
                            "index": idx,
                            "name": tool_name,
                            "args_preview": _safe_json(args),
                        },
                    )
                )
                result = self._call_tool_safely(tool_name, args)
                trace_records.append(
                    _build_trace_item(
                        step=f"工具执行完成：{tool_name}",
                        stage="tool_result",
                        payload={
                            "round": tool_rounds + 1,
                            "index": idx,
                            "name": tool_name,
                            "result_preview": _preview_text(result, MAX_TRACE_TEXT_CHARS),
                        },
                        level="error" if _is_tool_error(result) else "info",
                    )
                )
                working.append(ToolMessage(content=result, tool_call_id=call["id"]))

            tool_rounds += 1

        if tool_rounds >= self.settings.agent_tool_max_rounds:
            trace_records.append(
                _build_trace_item(
                    step=f"达到工具轮次上限（{self.settings.agent_tool_max_rounds}）",
                    stage="tool",
                    level="warning",
                )
            )

        return working

    def _call_tool_safely(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            return self._registry.call(name, arguments if isinstance(arguments, dict) else {})
        except Exception as exc:  # pylint: disable=broad-except
            return f"Tool error: {exc}"


def _build_chat_llm(settings: Settings, streaming: bool) -> ChatOpenAI:
    if not settings.agent_api_key:
        raise ValueError("agent api_key is not configured")

    sig = inspect.signature(ChatOpenAI)
    kwargs: dict[str, Any] = {"model": settings.agent_model}

    if "temperature" in sig.parameters:
        kwargs["temperature"] = settings.agent_temperature
    if "max_tokens" in sig.parameters:
        kwargs["max_tokens"] = settings.agent_max_tokens

    if "api_key" in sig.parameters:
        kwargs["api_key"] = settings.agent_api_key
    elif "openai_api_key" in sig.parameters:
        kwargs["openai_api_key"] = settings.agent_api_key

    if settings.agent_base_url:
        if "base_url" in sig.parameters:
            kwargs["base_url"] = settings.agent_base_url
        elif "openai_api_base" in sig.parameters:
            kwargs["openai_api_base"] = settings.agent_base_url

    if "streaming" in sig.parameters:
        kwargs["streaming"] = streaming

    llm = ChatOpenAI(**kwargs)
    if streaming and not getattr(llm, "streaming", False):
        try:
            llm.streaming = True
        except Exception:
            pass
    return llm


def _extract_tool_calls(message: AIMessage) -> list[dict[str, Any]]:
    raw_calls = getattr(message, "tool_calls", None) or []
    if not raw_calls:
        raw_calls = message.additional_kwargs.get("tool_calls", []) if message.additional_kwargs else []

    normalized: list[dict[str, Any]] = []
    for call in raw_calls:
        parsed = _normalize_tool_call(call)
        if parsed:
            normalized.append(parsed)
    return normalized


def _normalize_tool_call(call: Any) -> dict[str, Any] | None:
    if isinstance(call, dict):
        if "name" in call and "args" in call:
            return {
                "id": call.get("id") or str(uuid4()),
                "name": call.get("name", ""),
                "args": _parse_tool_args(call.get("args")),
            }
        if "function" in call:
            func = call.get("function") or {}
            return {
                "id": call.get("id") or str(uuid4()),
                "name": func.get("name", ""),
                "args": _parse_tool_args(func.get("arguments")),
            }
    name = getattr(call, "name", None)
    if name:
        return {
            "id": getattr(call, "id", None) or str(uuid4()),
            "name": name,
            "args": _parse_tool_args(getattr(call, "args", None)),
        }
    return None


def _parse_tool_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _chunk_to_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if content is None and hasattr(chunk, "text"):
        content = chunk.text
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _preview_text(value: Any, limit: int = MAX_TRACE_TEXT_CHARS) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated)"


def _safe_json(value: Any, limit: int = MAX_TRACE_TEXT_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return _preview_text(text, limit=limit)


def _is_tool_error(result: str) -> bool:
    return str(result).strip().lower().startswith("tool error:")


def _build_trace_item(
    step: str,
    stage: str,
    level: str = "info",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "step": _preview_text(step),
        "stage": stage,
        "level": level,
        "time": datetime.utcnow().isoformat() + "Z",
    }
    if payload:
        item["payload"] = payload
    return item


def _truncate_title(text: str, limit: int = 32) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"
