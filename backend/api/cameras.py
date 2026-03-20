from fastapi import APIRouter, HTTPException

from backend.models.camera import CameraInfo
from backend.vision.camera import camera_manager

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("")
async def list_cameras() -> list[CameraInfo]:
    devices = camera_manager.list_devices()
    return [CameraInfo(id=str(d.index), name=d.name, available=d.available) for d in devices]


@router.post("/{camera_id}/open")
async def open_camera(camera_id: str, width: int = 640, height: int = 480) -> dict:
    index = int(camera_id)
    if not camera_manager.open(index, width, height):
        raise HTTPException(status_code=500, detail=f"Failed to open camera {camera_id}")
    return {"status": "opened", "camera_id": camera_id}


@router.post("/{camera_id}/close")
async def close_camera(camera_id: str) -> dict:
    camera_manager.close(int(camera_id))
    return {"status": "closed", "camera_id": camera_id}


@router.get("/{camera_id}/frame")
async def get_frame(camera_id: str) -> dict:
    """Get a single frame as base64-encoded JPEG (for debugging)."""
    import base64

    jpeg = camera_manager.capture_frame_jpeg(int(camera_id))
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No frame available. Is camera open?")
    return {"camera_id": camera_id, "frame_b64": base64.b64encode(jpeg).decode()}
