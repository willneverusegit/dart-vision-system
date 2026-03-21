"""Detection API endpoints for starting/stopping dart hit detection."""

import asyncio
import base64
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.websocket import ws_manager
from backend.models.board import BoardModel
from backend.models.game import HitEvent
from backend.vision.detection import BackgroundModel
from backend.vision.pipeline import DetectionLoop

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detection", tags=["detection"])

# Module-level state
_background_model = BackgroundModel()
_detection_loop: DetectionLoop | None = None
_event_loop: asyncio.AbstractEventLoop | None = None


class DetectionStartRequest(BaseModel):
    camera_id: int = 0
    debug: bool = False


class DetectionStatus(BaseModel):
    running: bool = False
    camera_id: int | None = None
    has_background: bool = False


def _on_hit(hit: HitEvent, debug_jpeg: bytes | None) -> None:
    """Callback from DetectionLoop (runs in background thread)."""
    loop = _event_loop
    if loop is None:
        return

    async def _broadcast() -> None:
        await ws_manager.broadcast({"type": "hit_event", "data": hit.model_dump(mode="json")})
        if debug_jpeg is not None:
            b64 = base64.b64encode(debug_jpeg).decode("ascii")
            await ws_manager.broadcast({"type": "debug_frame", "data": b64})

    asyncio.run_coroutine_threadsafe(_broadcast(), loop)


@router.post("/background")
async def capture_background(camera_id: int = 0) -> dict:
    """Capture a background frame from the specified camera."""
    success = _background_model.capture_background(camera_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to capture background")
    return {"status": "ok", "camera_id": camera_id}


@router.post("/start")
async def start_detection(request: DetectionStartRequest) -> DetectionStatus:
    """Start the detection loop for a camera."""
    global _detection_loop, _event_loop  # noqa: PLW0603

    if _detection_loop and _detection_loop.is_running:
        raise HTTPException(status_code=409, detail="Detection already running")

    if _background_model.get_background() is None:
        raise HTTPException(
            status_code=400,
            detail="No background captured. Call /api/detection/background first.",
        )

    # Store the event loop for thread-safe broadcasting
    _event_loop = asyncio.get_running_loop()

    # Try to load board model from active profile
    board = None
    try:
        from backend.main import get_active_profile

        profile = get_active_profile()
        if profile and profile.homography:
            board = BoardModel(homography=profile.homography)
    except Exception:
        logger.warning("Could not load board model from active profile")

    _detection_loop = DetectionLoop(
        camera_id=request.camera_id,
        background_model=_background_model,
        board_model=board,
        on_hit=_on_hit,
        debug=request.debug,
    )
    _detection_loop.start()

    return DetectionStatus(
        running=True,
        camera_id=request.camera_id,
        has_background=True,
    )


@router.post("/stop")
async def stop_detection() -> DetectionStatus:
    """Stop the detection loop."""
    global _detection_loop  # noqa: PLW0603

    if _detection_loop:
        _detection_loop.stop()
        _detection_loop = None

    return DetectionStatus(
        running=False,
        has_background=_background_model.get_background() is not None,
    )


@router.get("/status")
async def get_detection_status() -> DetectionStatus:
    """Check if detection is running."""
    return DetectionStatus(
        running=_detection_loop.is_running if _detection_loop else False,
        camera_id=None,
        has_background=_background_model.get_background() is not None,
    )
