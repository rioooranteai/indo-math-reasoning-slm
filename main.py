"""Indo Math web app: FastAPI backend that also serves the dashboard.

Run from the project folder:
    uvicorn main:app --port 8000
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api import inference, pages
from core.config import PROJECT_DIR, settings
from services.model_service import model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load the model once, before the server accepts requests.
    settings.validate_model_paths()
    model_service.load()
    yield


app = FastAPI(title="Indo Math", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PROJECT_DIR / "static"), name="static")
app.include_router(inference.router)
app.include_router(pages.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port)
