"""
Base calibration functionality and shared utilities.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

from src.core import CalibrationData, BoardGeometry

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of a calibration operation."""
    success: bool
    data: Optional[CalibrationData]
    message: str
    warped_image: Optional[np.ndarray] = None  # Warped/corrected image for verification


class CalibratorBase:
    """
    Base class for calibration methods.

    Provides common functionality:
    - Homography computation
    - Image warping
    - Scaling factor calculation
    - Board detection utilities
    """

    def __init__(self, board_geometry: Optional[BoardGeometry] = None):
        """
        Initialize calibrator.

        Args:
            board_geometry: Dartboard geometry (default: BoardGeometry())
        """
        self.board_geometry = board_geometry or BoardGeometry()

    def compute_homography(
            self,
            src_points: np.ndarray,
            dst_points: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Compute homography matrix from point correspondences.

        Args:
            src_points: Source points (N, 2) in image
            dst_points: Destination points (N, 2) in canonical view

        Returns:
            (success, homography_matrix or None)
        """
        if len(src_points) < 4:
            logger.error("Need at least 4 points for homography")
            return False, None

        try:
            # Compute homography with RANSAC for robustness
            H, mask = cv2.findHomography(
                src_points,
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )

            if H is None:
                logger.error("Failed to compute homography")
                return False, None

            # Check for degenerate matrix
            if np.linalg.det(H) < 1e-6:
                logger.error("Degenerate homography matrix")
                return False, None

            inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 1.0
            logger.info(f"Homography computed (inlier ratio: {inlier_ratio:.2f})")

            return True, H

        except Exception as e:
            logger.error(f"Homography computation failed: {e}")
            return False, None

    def warp_image(
            self,
            image: np.ndarray,
            homography: np.ndarray,
            output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply perspective transformation to image.

        Args:
            image: Input image
            homography: 3x3 homography matrix
            output_size: (width, height) of output

        Returns:
            Warped image
        """
        return cv2.warpPerspective(
            image,
            homography,
            output_size,
            flags=cv2.INTER_LINEAR
        )

    def calculate_mm_per_pixel(
            self,
            known_distance_mm: float,
            measured_distance_px: float
    ) -> float:
        """
        Calculate scaling factor from known distance.

        Args:
            known_distance_mm: Known distance in millimeters
            measured_distance_px: Measured distance in pixels

        Returns:
            Millimeters per pixel
        """
        if measured_distance_px <= 0:
            logger.error("Invalid pixel distance")
            return 0.0

        mm_per_px = known_distance_mm / measured_distance_px
        logger.info(f"Calculated scale: {mm_per_px:.4f} mm/px")
        return mm_per_px

    def detect_board_circle(
            self,
            image: np.ndarray,
            min_radius: int = 100,
            max_radius: int = 400
    ) -> Optional[Tuple[int, int, int]]:
        """
        Detect dartboard outer circle using Hough Circle Transform.

        Args:
            image: Input image (BGR or grayscale)
            min_radius: Minimum circle radius in pixels
            max_radius: Maximum circle radius in pixels

        Returns:
            (center_x, center_y, radius) or None if not found
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=gray.shape[0] // 4,
            param1=100,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is None or len(circles[0]) == 0:
            logger.warning("No circles detected")
            return None

        # Take the first (strongest) circle
        circle = circles[0][0]
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

        logger.info(f"Detected board circle: center=({x}, {y}), radius={r}")
        return (x, y, r)

    def create_canonical_board_points(
            self,
            center: Tuple[float, float],
            radius: float
    ) -> np.ndarray:
        """
        Create canonical board corner points for standard dartboard.

        Creates 4 points at cardinal directions (top, right, bottom, left)
        on the outer double ring.

        Args:
            center: Board center (x, y)
            radius: Board outer radius in pixels

        Returns:
            Array of shape (4, 2) with corner points
        """
        cx, cy = center

        # Points at 12, 3, 6, 9 o'clock (top, right, bottom, left)
        points = np.array([
            [cx, cy - radius],  # Top (12 o'clock)
            [cx + radius, cy],  # Right (3 o'clock)
            [cx, cy + radius],  # Bottom (6 o'clock)
            [cx - radius, cy],  # Left (9 o'clock)
        ], dtype=np.float32)

        return points

    def validate_calibration(
            self,
            calibration_data: CalibrationData
    ) -> Tuple[bool, str]:
        """
        Validate calibration data for consistency.

        Args:
            calibration_data: Calibration to validate

        Returns:
            (is_valid, message)
        """
        # Check homography matrix
        H = calibration_data.homography_matrix
        if H.shape != (3, 3):
            return False, "Invalid homography matrix shape"

        if np.linalg.det(H) < 1e-6:
            return False, "Degenerate homography matrix"

        # Check scaling
        if calibration_data.mm_per_pixel <= 0 or calibration_data.mm_per_pixel > 10:
            return False, f"Unrealistic scale: {calibration_data.mm_per_pixel:.4f} mm/px"

        # Check board radius
        if calibration_data.board_radius_px < 50 or calibration_data.board_radius_px > 1000:
            return False, f"Unrealistic board radius: {calibration_data.board_radius_px} px"

        return True, "Calibration valid"