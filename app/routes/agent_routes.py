from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import AgentMessage, AgentSession
from app.schemas import (
    AgentMessageResponse,
    AgentSessionCreateRequest,
    AgentSessionDetailResponse,
    AgentSessionResponse,
)
from app.services.agent_service import AgentService

router = APIRouter()


@router.get("/api/chat/agent/stream")
def stream_agent(
    message: str = Query(..., min_length=1),
    session_id: str | None = None,
) -> StreamingResponse:
    settings = get_settings()
    service = AgentService(settings)
    generator = service.stream_chat(message, session_id=session_id)
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(generator, media_type="text/event-stream", headers=headers)


@router.post("/api/agent/sessions", response_model=AgentSessionResponse)
def create_session(
    payload: AgentSessionCreateRequest | None = Body(default=None),
    db: Session = Depends(get_db),
) -> AgentSessionResponse:
    title = (payload.title if payload else None) or "New Chat"
    session = AgentSession(id=str(uuid4()), title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return AgentSessionResponse(
        session_id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/api/agent/sessions", response_model=list[AgentSessionResponse])
def list_sessions(
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[AgentSessionResponse]:
    sessions = (
        db.query(AgentSession)
        .order_by(AgentSession.updated_at.desc())
        .limit(limit)
        .all()
    )
    return [
        AgentSessionResponse(
            session_id=item.id,
            title=item.title,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in sessions
    ]


@router.get("/api/agent/sessions/{session_id}", response_model=AgentSessionDetailResponse)
def get_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> AgentSessionDetailResponse:
    session = db.query(AgentSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    messages = (
        db.query(AgentMessage)
        .filter_by(session_id=session_id)
        .order_by(AgentMessage.id.asc())
        .all()
    )

    return AgentSessionDetailResponse(
        session=AgentSessionResponse(
            session_id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
        ),
        messages=[
            AgentMessageResponse(
                id=item.id,
                role=item.role,
                content=item.content,
                status=item.status,
                created_at=item.created_at,
            )
            for item in messages
        ],
    )
