from __future__ import annotations

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import GraphBuildRequest, GraphBuildResponse
from app.services import graph_api_service
from app.services.service_errors import ServiceError


def build_graph(
    request: GraphBuildRequest, db: Session = Depends(get_db)
) -> GraphBuildResponse:
    try:
        return graph_api_service.build_graph(request, db)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
