from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.calibration import router as calibration_router
from backend.api.cameras import router as cameras_router
from backend.api.game import router as game_router
from backend.api.profiles import router as profiles_router
from backend.api.websocket import router as ws_router

# Loaded at startup; accessible via get_active_profile()
_active_profile = None


def get_active_profile():
    """Return the currently loaded CameraProfile, or None."""
    return _active_profile


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _active_profile  # noqa: PLW0603
    from backend.services.profile_store import get_last_profile

    _active_profile = get_last_profile()
    yield


app = FastAPI(title="Dart Vision API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calibration_router)
app.include_router(cameras_router)
app.include_router(game_router)
app.include_router(profiles_router)
app.include_router(ws_router)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


# Serve frontend static files (must be last, catches all unmatched routes)
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
