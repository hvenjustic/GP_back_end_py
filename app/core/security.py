from __future__ import annotations

import os


def get_api_key_header() -> str:
    return os.getenv("API_KEY_HEADER", "X-API-Key")
