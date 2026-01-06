from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from app.config import get_settings


def _bool_env(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def ensure_dependencies() -> None:
    settings = get_settings()
    if not settings.auto_install_deps:
        return

    root = Path(__file__).resolve().parent.parent.parent
    requirements = root / "requirements.txt"
    if not requirements.exists():
        return

    marker = root / ".deps_installed"
    req_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == req_hash:
        return

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )
    marker.write_text(req_hash, encoding="utf-8")


def ensure_playwright_browsers() -> None:
    settings = get_settings()
    if not settings.auto_install_playwright:
        return

    version = _get_package_version("playwright")
    if not version:
        return

    root = Path(__file__).resolve().parent.parent.parent
    marker = root / ".playwright_installed"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == version:
        return

    subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
    marker.write_text(version, encoding="utf-8")
