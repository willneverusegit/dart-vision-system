"""Board detection and homography computation."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard dartboard dimensions (mm)
BOARD_RADIUS_MM = 170.0  # outer double ring


def detect_board_circles(
    frame: np.ndarray,
    min_radius: int = 50,
    max_radius: int = 300,
) -> tuple[float, float, float] | None:
    """Detect the dartboard center and radius using Hough circles.

    Returns (center_x, center_y, radius) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=50,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        logger.warning("No circles detected in frame")
        return None

    # Take the largest circle as the board
    circles = np.round(circles[0]).astype(int)
    best = max(circles, key=lambda c: c[2])
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    logger.info("Detected board center=(%d, %d), radius=%d", cx, cy, r)
    return (float(cx), float(cy), float(r))


def compute_homography_from_points(
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
) -> np.ndarray | None:
    """Compute homography from source to destination points.

    Args:
        src_points: 4+ points in the camera image (pixel coordinates)
        dst_points: 4+ corresponding points in the normalized board space

    Returns:
        3x3 homography matrix or None on failure.
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        logger.error("Need at least 4 point pairs for homography")
        return None

    src = np.array(src_points, dtype=np.float64)
    dst = np.array(dst_points, dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        logger.error("Homography computation failed")
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    logger.info("Homography computed with %d/%d inliers", inliers, len(src_points))
    return H


def compute_homography_from_circle(
    center: tuple[float, float],
    radius: float,
    output_size: int = 500,
) -> np.ndarray:
    """Compute a simple homography that maps the detected circle to a centered square.

    This assumes the camera is roughly perpendicular to the board.
    For angled cameras, use the 4-point method instead.
    """
    cx, cy, r = center[0], center[1], radius

    # Map the bounding box of the circle to the output square
    src = np.array(
        [
            [cx - r, cy - r],
            [cx + r, cy - r],
            [cx + r, cy + r],
            [cx - r, cy + r],
        ],
        dtype=np.float64,
    )

    dst = np.array(
        [
            [0, 0],
            [output_size, 0],
            [output_size, output_size],
            [0, output_size],
        ],
        dtype=np.float64,
    )

    H, _ = cv2.findHomography(src, dst)
    return H


def warp_to_topdown(
    frame: np.ndarray,
    homography: np.ndarray,
    output_size: int = 500,
) -> np.ndarray:
    """Warp a camera frame to a top-down view using the homography."""
    return cv2.warpPerspective(frame, homography, (output_size, output_size))


def pixel_to_board(
    pixel_point: tuple[float, float],
    homography: np.ndarray,
    output_size: int = 500,
) -> tuple[float, float]:
    """Transform a pixel coordinate to board coordinates (mm from center).

    Returns (x_mm, y_mm) where (0, 0) is the board center.
    """
    pt = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(pt, homography)
    tx, ty = transformed[0, 0]

    # Map from output_size coordinates to mm
    center = output_size / 2
    scale = BOARD_RADIUS_MM / (output_size / 2)
    x_mm = (tx - center) * scale
    y_mm = (ty - center) * scale
    return (x_mm, y_mm)
