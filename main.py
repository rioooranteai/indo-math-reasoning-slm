import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.config import settings, validate_paths
from services.model_service import model_service
from api.routes import router as api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

# Root project (folder yang berisi main.py). static/ dan templates/
# berada satu folder dengan kode backend - ini yang membuat frontend
# & backend "menyatu".
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: validasi path model lalu muat model sekali saja,
    # sebelum server mulai menerima request.
    validate_paths()
    model_service.load()
    yield
    # Shutdown: tidak ada resource khusus yang perlu dibersihkan.


app = FastAPI(
    title="Dashboard Inferensi Model Matematika",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Mount frontend ---
# "/static" adalah URL prefix, STATIC_DIR adalah folder fisiknya.
# Setelah ini, static/css/style.css bisa diakses lewat
# GET /static/css/style.css, dan static/js/dashboard.js lewat
# GET /static/js/dashboard.js. Ini yang dimaksud "mount" di FastAPI:
# menempelkan sebuah sub-aplikasi (di sini: file server statis) pada
# path tertentu di aplikasi utama.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Routes API ---
app.include_router(api_router)


# --- Route halaman utama ---
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
