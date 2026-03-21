"""Board geometry, field mapping, and score calculation."""

import math

import cv2
import numpy as np

from backend.models.board import BoardModel

# Standard dartboard sector layout (clockwise from top, 20 at 12 o'clock)
SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
SECTOR_WIDTH_DEG = 360.0 / 20  # 18 degrees per sector


def pixel_to_board(
    pixel_point: tuple[float, float],
    homography: np.ndarray,
    output_size: int = 500,
) -> tuple[float, float]:
    """Transform a pixel coordinate to board coordinates (mm from center).

    Delegates to board_fitting.pixel_to_board for the actual transform.
    """
    from backend.vision.board_fitting import BOARD_RADIUS_MM

    pt = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(pt, homography)
    tx, ty = transformed[0, 0]

    center = output_size / 2
    scale = BOARD_RADIUS_MM / (output_size / 2)
    x_mm = (tx - center) * scale
    y_mm = (ty - center) * scale
    return (x_mm, y_mm)


def board_to_polar(board_point: tuple[float, float]) -> tuple[float, float]:
    """Convert board coordinates (mm from center) to polar (radius_mm, angle_deg).

    Angle is measured clockwise from 12 o'clock (top), 0-360.
    """
    x, y = board_point
    radius = math.sqrt(x * x + y * y)
    # atan2 gives angle from positive x-axis, counter-clockwise
    # We want clockwise from top (negative y-axis)
    angle_rad = math.atan2(x, -y)  # x/(-y) gives clockwise from top
    angle_deg = math.degrees(angle_rad) % 360
    return (radius, angle_deg)


def polar_to_field(
    radius: float,
    angle: float,
    board: BoardModel,
) -> str:
    """Map polar coordinates to a dartboard field name.

    Returns field names like "T20", "D16", "BULL", "25", "S5", "MISS".
    """
    radii = board.ring_radii

    # Bull's eye
    if radius <= radii["bull"]:
        return "BULL"

    # Outer bull (single bull / 25)
    if radius <= radii["outer_bull"]:
        return "25"

    # Miss (outside double ring)
    if radius > radii["double_outer"]:
        return "MISS"

    # Determine sector number
    sectors = board.sector_angles if board.sector_angles else SECTOR_ORDER
    half_sector = SECTOR_WIDTH_DEG / 2  # 9 degrees

    # Shift angle so sector 20 is centered at 0 degrees
    shifted = (angle + half_sector) % 360
    sector_index = int(shifted / SECTOR_WIDTH_DEG) % 20
    number = sectors[sector_index]

    # Determine ring
    if radius <= radii["triple_inner"]:
        return f"S{number}"  # inner single
    if radius <= radii["triple_outer"]:
        return f"T{number}"  # triple
    if radius <= radii["double_inner"]:
        return f"S{number}"  # outer single
    # Must be double
    return f"D{number}"


def get_score(field: str) -> tuple[int, int]:
    """Convert a field name to (base_score, multiplier).

    Examples:
        "T20" -> (20, 3)
        "D16" -> (16, 2)
        "S5"  -> (5, 1)
        "BULL" -> (50, 1)
        "25"  -> (25, 1)
        "MISS" -> (0, 0)
    """
    if field == "BULL":
        return (50, 1)
    if field == "25":
        return (25, 1)
    if field == "MISS":
        return (0, 0)

    if len(field) < 2:
        return (0, 0)

    prefix = field[0]
    try:
        number = int(field[1:])
    except ValueError:
        return (0, 0)

    if prefix == "T":
        return (number, 3)
    if prefix == "D":
        return (number, 2)
    if prefix == "S":
        return (number, 1)

    return (0, 0)
