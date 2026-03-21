"""REST endpoints for warp (top-down view) configuration and frame retrieval."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from backend.vision.camera import camera_manager
from backend.vision.warp import warp_engine

router = APIRouter(prefix="/api/warp", tags=["warp"])


class WarpStatusResponse(BaseModel):
    """Response model for warp status."""

    configured: bool
    output_size: int


class WarpConfigureRequest(BaseModel):
    """Request model for configuring the warp homography."""

    homography: list[list[float]]
    output_size: int = 500


class WarpConfigureResponse(BaseModel):
    """Response model after successful warp configuration."""

    configured: bool
    output_size: int


@router.get("/status")
def warp_status() -> WarpStatusResponse:
    """Return whether the warp engine is configured."""
    return WarpStatusResponse(
        configured=warp_engine.is_configured(),
        output_size=warp_engine._output_size,
    )


@router.post("/configure")
def warp_configure(req: WarpConfigureRequest) -> WarpConfigureResponse:
    """Configure the warp engine with a 3x3 homography matrix.

    Args:
        req: Request containing homography (3x3 nested list) and output_size.
    """
    if len(req.homography) != 3 or any(len(row) != 3 for row in req.homography):
        raise HTTPException(status_code=422, detail="Homography must be a 3x3 matrix")
    warp_engine.set_homography(req.homography, req.output_size)
    return WarpConfigureResponse(configured=True, output_size=req.output_size)


@router.get("/frame/{camera_id}")
def warp_frame(camera_id: str, quality: int = 80) -> Response:
    """Capture a frame from the given camera, warp it, and return as JPEG.

    Args:
        camera_id: Camera index as string.
        quality: JPEG quality (0-100).
    """
    if not warp_engine.is_configured():
        raise HTTPException(status_code=400, detail="Warp engine not configured")

    index = int(camera_id)
    frame = camera_manager.capture_frame(index)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"No frame from camera {camera_id}")

    jpeg_bytes = warp_engine.warp_frame_jpeg(frame, quality)
    return Response(content=jpeg_bytes, media_type="image/jpeg")
