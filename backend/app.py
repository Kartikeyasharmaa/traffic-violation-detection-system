from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.database import init_db
from backend.routes.detectors import router as detectors_router
from backend.routes.violations import router as violations_router
from config import settings


app = FastAPI(title=settings.project_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(violations_router)
app.include_router(detectors_router)
app.mount("/images", StaticFiles(directory=str(settings.image_dir)), name="images")
app.mount("/frontend", StaticFiles(directory=str(settings.frontend_dir)), name="frontend")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard_alias() -> RedirectResponse:
    return RedirectResponse(url="/")


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
