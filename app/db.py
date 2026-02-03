from __future__ import annotations

import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)


def _build_connect_args() -> dict:
    if settings.database_url.startswith("mysql+pymysql"):
        timeout = settings.database_connect_timeout
        return {
            "connect_timeout": timeout,
            "read_timeout": timeout,
            "write_timeout": timeout,
        }
    return {}


_engine_kwargs = {"pool_pre_ping": True}
_connect_args = _build_connect_args()
if _connect_args:
    _engine_kwargs["connect_args"] = _connect_args

engine = create_engine(settings.database_url, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    from app import models

    try:
        models.Base.metadata.create_all(bind=engine)
        _ensure_site_tasks_columns()
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "timed out" in lowered or "timeout" in lowered:
            hint = f"数据库连接超时（{settings.database_connect_timeout}s）"
        else:
            hint = "数据库初始化失败"
        logger.error("%s：%s", hint, message, exc_info=True)
        raise RuntimeError(f"{hint}，请检查数据库服务是否可用。原始错误：{message}") from exc


def _ensure_site_tasks_columns() -> None:
    inspector = inspect(engine)
    if "site_tasks" not in inspector.get_table_names():
        return
    columns = {col["name"] for col in inspector.get_columns("site_tasks")}
    if "on_sale" not in columns:
        logger.warning("site_tasks.on_sale 缺失，正在补齐字段")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE site_tasks "
                    "ADD COLUMN on_sale BOOLEAN NOT NULL DEFAULT 0"
                )
            )
    if "embedding" not in columns:
        logger.warning("site_tasks.embedding 缺失，正在补齐字段")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE site_tasks "
                    "ADD COLUMN embedding JSON"
                )
            )
    if "coord_3d" not in columns:
        logger.warning("site_tasks.coord_3d 缺失，正在补齐字段")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE site_tasks "
                    "ADD COLUMN coord_3d JSON"
                )
            )
    if "embedding_updated_at" not in columns:
        logger.warning("site_tasks.embedding_updated_at 缺失，正在补齐字段")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE site_tasks "
                    "ADD COLUMN embedding_updated_at DATETIME"
                )
            )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
