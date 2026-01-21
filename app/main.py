from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from app.core.bootstrap import ensure_dependencies, ensure_playwright_browsers
from app.core.logger import configure_logging

ensure_dependencies()
ensure_playwright_browsers()
configure_logging()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.agent_routes import router as agent_router
from app.routes.api_routes import router as api_router
from app.routes.crawl_routes import router as crawl_router
from app.routes.graph_routes import router as graph_router
from app.config import get_settings
from app.db import init_db

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Site Crawl Service", lifespan=lifespan)

if settings.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(crawl_router)
app.include_router(graph_router)
app.include_router(agent_router)
app.include_router(api_router)

if __name__ == "__main__":
    import subprocess
    import sys

    import uvicorn

    def _start_worker() -> subprocess.Popen:
        worker_concurrency = str(settings.worker_concurrency)
        log_file = settings.worker_log_file
        if not os.path.isabs(log_file):
            log_file = str(Path(__file__).resolve().parent.parent / log_file)
        cmd = [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "app.services.crawl_tasks",
            "worker",
            "--loglevel=info",
            f"--concurrency={worker_concurrency}",
            f"--logfile={log_file}",
        ]
        logger.info("starting worker: %s", " ".join(cmd))
        return subprocess.Popen(cmd)

    def _stop_worker(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        logger.info("stopping worker pid=%s", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    worker = _start_worker()
    try:
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    finally:
        _stop_worker(worker)
