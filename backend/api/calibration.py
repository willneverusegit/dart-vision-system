from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.models.camera import CameraProfile, CameraRole, Roi
from backend.vision.calibration import (
    CalibrationSession,
    list_profiles,
    load_profile,
    save_profile,
)

router = APIRouter(prefix="/api/calibrate", tags=["calibration"])

# Active calibration sessions (one per camera)
_sessions: dict[str, CalibrationSession] = {}


class CalibrationStartRequest(BaseModel):
    camera_id: str
    role: CameraRole = CameraRole.LEFT


class CalibrationStatus(BaseModel):
    camera_id: str
    frames_collected: int
    ready: bool
    reprojection_error: float | None = None


class RoiRequest(BaseModel):
    camera_id: str
    x: int
    y: int
    w: int
    h: int


@router.post("/start")
async def start_calibration(req: CalibrationStartRequest) -> CalibrationStatus:
    session = CalibrationSession(req.camera_id, req.role)
    _sessions[req.camera_id] = session
    return CalibrationStatus(
        camera_id=req.camera_id,
        frames_collected=0,
        ready=False,
    )


@router.get("/status/{camera_id}")
async def get_calibration_status(camera_id: str) -> CalibrationStatus:
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No calibration session for this camera")
    return CalibrationStatus(
        camera_id=camera_id,
        frames_collected=session.frame_count,
        ready=session.frame_count >= 15,
        reprojection_error=session.reprojection_error,
    )


@router.post("/finish")
async def finish_calibration(camera_id: str, profile_name: str | None = None) -> CameraProfile:
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No calibration session for this camera")

    error = session.calibrate()
    if error is None:
        raise HTTPException(status_code=400, detail="Calibration failed — not enough frames")

    profile = session.get_profile()
    save_profile(profile, profile_name)
    del _sessions[camera_id]
    return profile


@router.post("/roi")
async def set_roi(req: RoiRequest) -> dict:
    """Store ROI for a camera profile."""
    # Load existing profile or create minimal one
    profiles = list_profiles()
    matching = [p for p in profiles if req.camera_id in p]

    if matching:
        profile = load_profile(matching[0])
        if profile:
            profile.roi = Roi(x=req.x, y=req.y, w=req.w, h=req.h)
            save_profile(profile, matching[0])
            return {"status": "roi_saved", "camera_id": req.camera_id, "roi": req.model_dump()}

    return {"status": "no_profile_found", "camera_id": req.camera_id}


@router.get("/profiles")
async def get_profiles() -> list[str]:
    return list_profiles()


@router.get("/profiles/{name}")
async def get_profile(name: str) -> CameraProfile:
    profile = load_profile(name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    return profile
