from __future__ import annotations

from fastapi import Body, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.schemas import (
    AgentSessionCreateRequest,
    AgentSessionDetailResponse,
    AgentSessionResponse,
)
from app.services.agent_service import AgentService
from app.services.agent_session_service import (
    create_session as create_agent_session,
    get_session_detail,
    list_sessions as list_agent_sessions,
)
from app.services.service_errors import ServiceError


def stream_agent(
    message: str = Query(..., min_length=1),
    session_id: str | None = None,
) -> StreamingResponse:
    settings = get_settings()
    service = AgentService(settings)
    generator = service.stream_chat(message, session_id=session_id)
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(generator, media_type="text/event-stream", headers=headers)


def create_session(
    payload: AgentSessionCreateRequest | None = Body(default=None),
    db: Session = Depends(get_db),
) -> AgentSessionResponse:
    try:
        return create_agent_session(payload, db)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def list_sessions(
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[AgentSessionResponse]:
    return list_agent_sessions(limit, db)


def get_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> AgentSessionDetailResponse:
    try:
        return get_session_detail(session_id, db)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
