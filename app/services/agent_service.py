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

                content_parts: list[str] = []

                prepared_messages = self._prepare_messages_for_stream(messages)
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
                assistant_message.content = final_text
                assistant_message.status = "DONE"
                session.updated_at = datetime.utcnow()
                db.commit()

                yield _sse_event(
                    "done",
                    {
                        "messageId": str(assistant_message.id),
                        "sessionId": session.id,
                    },
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("agent stream failed")
                error_text = f"Agent error: {exc}"
                if assistant_message:
                    assistant_message.content = error_text
                    assistant_message.status = "FAILED"
                    db.commit()
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

    def _prepare_messages_for_stream(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        enabled = {name.strip() for name in self.settings.agent_tools_enabled if name.strip()}
        if not (self.settings.agent_use_tools and enabled):
            return messages

        tools = self._registry.to_langchain_tools(enabled)
        if not tools:
            return messages

        llm = self._get_llm(streaming=False)
        llm_with_tools = llm.bind_tools(tools)
        tool_rounds = 0
        working = list(messages)

        while tool_rounds < self.settings.agent_tool_max_rounds:
            response = llm_with_tools.invoke(working)
            if not isinstance(response, AIMessage):
                break
            tool_calls = _extract_tool_calls(response)
            if not tool_calls:
                break

            working.append(response)
            for call in tool_calls:
                result = self._call_tool_safely(call["name"], call["args"])
                working.append(ToolMessage(content=result, tool_call_id=call["id"]))

            tool_rounds += 1

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


def _truncate_title(text: str, limit: int = 32) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"
