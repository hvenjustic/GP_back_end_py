from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import Any, Iterator
from uuid import uuid4

from openai import OpenAI
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import SessionLocal
from app.models import AgentMessage, AgentSession
from app.services.agent_tools import ToolRegistry, build_default_registry

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(self, settings: Settings, registry: ToolRegistry | None = None) -> None:
        self.settings = settings
        self._client: OpenAI | None = None
        self._registry = registry or build_default_registry(settings)

    def stream_chat(self, message: str, session_id: str | None = None) -> Iterator[str]:
        def generator() -> Iterator[str]:
            db = SessionLocal()
            assistant_message: AgentMessage | None = None
            session: AgentSession | None = None
            try:
                session = self._get_or_create_session(db, session_id, message)
                user_message = self._add_message(db, session.id, "user", message)
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
                client = self._get_client()

                prepared_messages = self._prepare_messages_for_stream(client, messages)

                stream = client.chat.completions.create(
                    model=self.settings.agent_model,
                    messages=prepared_messages,
                    temperature=self.settings.agent_temperature,
                    max_tokens=self.settings.agent_max_tokens,
                    stream=True,
                )

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content or ""
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

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.agent_api_key:
                raise ValueError("agent api_key is not configured")
            client_kwargs: dict[str, Any] = {"api_key": self.settings.agent_api_key}
            if self.settings.agent_base_url:
                client_kwargs["base_url"] = self.settings.agent_base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

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

    def _build_messages(self, db: Session, session_id: str) -> list[dict[str, Any]]:
        query = (
            db.query(AgentMessage)
            .filter_by(session_id=session_id)
            .order_by(AgentMessage.id.desc())
            .limit(self.settings.agent_max_history)
        )
        items = list(reversed(query.all()))

        messages: list[dict[str, Any]] = []
        if self.settings.agent_system_prompt:
            messages.append({"role": "system", "content": self.settings.agent_system_prompt})

        for item in items:
            if item.role not in {"user", "assistant"}:
                continue
            messages.append({"role": item.role, "content": item.content or ""})
        return messages

    def _prepare_messages_for_stream(
        self, client: OpenAI, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        enabled = {name.strip() for name in self.settings.agent_tools_enabled if name.strip()}
        if not (self.settings.agent_use_tools and enabled):
            return messages

        tools = self._registry.to_openai_tools(enabled)
        tool_rounds = 0
        working = list(messages)

        while tool_rounds < self.settings.agent_tool_max_rounds:
            response = client.chat.completions.create(
                model=self.settings.agent_model,
                messages=working,
                tools=tools,
                temperature=self.settings.agent_temperature,
                max_tokens=self.settings.agent_max_tokens,
            )
            choice = response.choices[0]
            tool_calls = choice.message.tool_calls or []
            if not tool_calls:
                break

            working.append(
                {
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            for call in tool_calls:
                result = self._call_tool_safely(call.function.name, call.function.arguments)
                working.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

            tool_rounds += 1

        return working

    def _call_tool_safely(self, name: str, arguments: str | None) -> str:
        try:
            payload = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"Tool error: invalid arguments ({exc})"
        try:
            return self._registry.call(name, payload if isinstance(payload, dict) else {})
        except Exception as exc:  # pylint: disable=broad-except
            return f"Tool error: {exc}"


def _truncate_title(text: str, limit: int = 32) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"
