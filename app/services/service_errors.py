from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServiceError(Exception):
    status_code: int
    message: str
