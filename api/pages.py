"""Server-rendered HTML pages of the dashboard."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.config import PROJECT_DIR

templates = Jinja2Templates(directory=PROJECT_DIR / "templates")

router = APIRouter(include_in_schema=False, default_response_class=HTMLResponse)


def _render(request: Request, template: str, page: str) -> HTMLResponse:
    # `page` marks the active menu item and selects the page layout.
    return templates.TemplateResponse(request, template, {"page": page})


@router.get("/")
def solve_page(request: Request) -> HTMLResponse:
    return _render(request, "index.html", "solve")


@router.get("/banyak-soal")
def batch_page(request: Request) -> HTMLResponse:
    return _render(request, "batch.html", "batch")


@router.get("/tentang")
def about_page(request: Request) -> HTMLResponse:
    return _render(request, "about.html", "about")
