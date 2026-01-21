from __future__ import annotations

from uuid import uuid4

from sqlalchemy.orm import Session

from app.models import AgentMessage, AgentSession
from app.schemas import (
    AgentMessageResponse,
    AgentSessionCreateRequest,
    AgentSessionDetailResponse,
    AgentSessionResponse,
)
from app.services.service_errors import ServiceError


def create_session(
    payload: AgentSessionCreateRequest | None, db: Session
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


def list_sessions(limit: int, db: Session) -> list[AgentSessionResponse]:
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


def get_session_detail(session_id: str, db: Session) -> AgentSessionDetailResponse:
    session = db.query(AgentSession).filter_by(id=session_id).first()
    if not session:
        raise ServiceError(status_code=404, message="session not found")

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
