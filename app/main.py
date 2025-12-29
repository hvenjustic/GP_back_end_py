from fastapi import FastAPI

from app.api import router
from app.db import init_db

app = FastAPI(title="Site Crawl Service")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.include_router(router)

