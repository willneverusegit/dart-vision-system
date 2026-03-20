from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class CameraRole(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"


class CameraInfo(BaseModel):
    """Basic camera device information."""

    id: str
    name: str
    available: bool = True


class Intrinsics(BaseModel):
    """Camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: list[float]


class Roi(BaseModel):
    """Region of interest rectangle."""

    x: int
    y: int
    w: int
    h: int


class CameraProfile(BaseModel):
    """Complete camera calibration profile."""

    id: str
    role: CameraRole
    resolution: tuple[int, int]
    intrinsics: Intrinsics | None = None
    roi: Roi | None = None
    homography: list[list[float]] | None = None
    timestamp: datetime | None = None
