from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.database import init_db
from backend.routes.auth import router as auth_router
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

app.include_router(auth_router)
app.include_router(violations_router)
app.include_router(detectors_router)
app.mount("/images", StaticFiles(directory=str(settings.image_dir)), name="images")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def _frontend_index_path() -> Path:
    built_index = settings.frontend_dist_dir / "index.html"
    if built_index.exists():
        return built_index
    return settings.frontend_dir / "index.html"


@app.get("/assets/{asset_path:path}", include_in_schema=False)
def frontend_assets(asset_path: str) -> FileResponse:
    asset_file = settings.frontend_dist_dir / "assets" / asset_path
    if not asset_file.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(asset_file)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(_frontend_index_path())


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
