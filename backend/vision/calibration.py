"""Camera calibration using ChArUco boards."""

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

from backend.models.camera import CameraProfile, CameraRole, Intrinsics

logger = logging.getLogger(__name__)

# ChArUco board configuration
CHARUCO_DICT = aruco.DICT_6X6_250
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH = 0.04  # meters
CHARUCO_MARKER_LENGTH = 0.02  # meters

PROFILES_DIR = Path(__file__).parent.parent / "data" / "profiles"


def get_charuco_board() -> aruco.CharucoBoard:
    """Create a ChArUco board for calibration."""
    dictionary = aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_MARKER_LENGTH,
        dictionary,
    )
    return board


def get_aruco_dictionary():
    """Get the ArUco dictionary used for calibration."""
    return aruco.getPredefinedDictionary(CHARUCO_DICT)


class CalibrationSession:
    """Manages the calibration process for a single camera."""

    def __init__(self, camera_id: str, role: CameraRole = CameraRole.LEFT) -> None:
        self.camera_id = camera_id
        self.role = role
        self.board = get_charuco_board()
        self.dictionary = get_aruco_dictionary()
        self.detector_params = aruco.DetectorParameters()

        self.all_charuco_corners: list[np.ndarray] = []
        self.all_charuco_ids: list[np.ndarray] = []
        self.image_size: tuple[int, int] | None = None

        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.reprojection_error: float | None = None

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a frame for marker detection.

        Returns dict with detection results.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.image_size = gray.shape[::-1]  # (width, height)

        # Detect ArUco markers
        corners, ids, rejected = aruco.detectMarkers(
            gray, self.dictionary, parameters=self.detector_params
        )

        result = {
            "markers_detected": 0,
            "charuco_corners": 0,
            "frames_collected": len(self.all_charuco_corners),
            "ready": len(self.all_charuco_corners) >= 15,
        }

        if ids is None or len(ids) < 4:
            return result

        result["markers_detected"] = len(ids)

        # Interpolate ChArUco corners
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board
        )

        if retval is not None and retval >= 6:
            self.all_charuco_corners.append(charuco_corners)
            self.all_charuco_ids.append(charuco_ids)
            result["charuco_corners"] = retval
            result["frames_collected"] = len(self.all_charuco_corners)
            result["ready"] = len(self.all_charuco_corners) >= 15

        return result

    def calibrate(self) -> float | None:
        """Run calibration from collected frames.

        Returns reprojection error or None on failure.
        """
        if len(self.all_charuco_corners) < 15:
            logger.error(
                "Not enough frames for calibration: %d (need >= 15)",
                len(self.all_charuco_corners),
            )
            return None

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=self.all_charuco_corners,
            imagePoints=self.all_charuco_ids,
            imageSize=self.image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.reprojection_error = ret
        logger.info("Calibration complete. Reprojection error: %.4f", ret)
        return ret

    def get_profile(self, resolution: tuple[int, int] | None = None) -> CameraProfile:
        """Create a CameraProfile from calibration results."""
        intrinsics = None
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            intrinsics = Intrinsics(
                fx=float(self.camera_matrix[0, 0]),
                fy=float(self.camera_matrix[1, 1]),
                cx=float(self.camera_matrix[0, 2]),
                cy=float(self.camera_matrix[1, 2]),
                dist_coeffs=self.dist_coeffs.flatten().tolist(),
            )

        return CameraProfile(
            id=self.camera_id,
            role=self.role,
            resolution=resolution or self.image_size or (640, 480),
            intrinsics=intrinsics,
            timestamp=datetime.now(),
        )

    @property
    def frame_count(self) -> int:
        return len(self.all_charuco_corners)


def save_profile(profile: CameraProfile, name: str | None = None) -> Path:
    """Save a camera profile to JSON file."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    filename = name or f"profile_{profile.id}_{profile.role}"
    filepath = PROFILES_DIR / f"{filename}.json"
    filepath.write_text(profile.model_dump_json(indent=2))
    logger.info("Saved profile to %s", filepath)
    return filepath


def load_profile(name: str) -> CameraProfile | None:
    """Load a camera profile from JSON file."""
    filepath = PROFILES_DIR / f"{name}.json"
    if not filepath.exists():
        return None
    data = json.loads(filepath.read_text())
    return CameraProfile.model_validate(data)


def list_profiles() -> list[str]:
    """List all saved profile names."""
    if not PROFILES_DIR.exists():
        return []
    return [p.stem for p in PROFILES_DIR.glob("*.json")]
