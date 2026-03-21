"""Confidence scoring for dart hit detection.

Produces a 0.0-1.0 confidence value from three weighted factors:
- Wire distance (40%): How far the hit is from the nearest wire boundary
- Detection quality (40%): Quality of the contour/tip extraction
- Triangulation (20%): Reprojection error from stereo (if available)
"""

import math

from backend.scoring.board_model import SECTOR_WIDTH_DEG

# Weight factors
WEIGHT_WIRE = 0.4
WEIGHT_DETECTION = 0.4
WEIGHT_TRIANGULATION = 0.2

# Review threshold
REVIEW_THRESHOLD = 0.6

# Wire boundaries (ring radii in mm)
RING_BOUNDARIES = [6.35, 15.9, 99.0, 107.0, 162.0, 170.0]

# Half-width of wire in mm (standard dart wire ~1.5mm diameter)
WIRE_HALF_WIDTH_MM = 0.75

# Maximum contour area considered "good" detection (pixels)
MAX_GOOD_CONTOUR_AREA = 5000
MIN_GOOD_CONTOUR_AREA = 200


def calculate_confidence(
    radius_mm: float,
    angle_deg: float,
    contour_area: float | None = None,
    contour_solidity: float | None = None,
    reprojection_error: float | None = None,
) -> float:
    """Calculate overall confidence for a dart hit.

    Args:
        radius_mm: Distance from board center in mm.
        angle_deg: Angle from top (clockwise) in degrees.
        contour_area: Area of the detected dart contour in pixels.
        contour_solidity: Solidity of the contour (area / convex hull area, 0-1).
        reprojection_error: Stereo triangulation reprojection error in pixels.

    Returns:
        Confidence value between 0.0 and 1.0.
    """
    wire_conf = wire_distance_confidence(radius_mm, angle_deg)
    det_conf = detection_quality_confidence(contour_area, contour_solidity)
    tri_conf = triangulation_confidence(reprojection_error)

    overall = (
        WEIGHT_WIRE * wire_conf + WEIGHT_DETECTION * det_conf + WEIGHT_TRIANGULATION * tri_conf
    )
    return round(max(0.0, min(1.0, overall)), 3)


def wire_distance_confidence(radius_mm: float, angle_deg: float) -> float:
    """Confidence based on distance to nearest wire.

    Wires exist at ring boundaries (concentric circles) and sector
    boundaries (radial lines every 18 degrees). The closer to a wire,
    the lower the confidence.

    Returns:
        0.0 (on wire) to 1.0 (far from any wire).
    """
    # Distance to nearest ring wire
    min_ring_dist = _min_ring_distance(radius_mm)

    # Distance to nearest sector wire (only relevant between outer_bull and double_outer)
    min_sector_dist = _min_sector_distance(radius_mm, angle_deg)

    # Use the smaller distance
    min_dist = min(min_ring_dist, min_sector_dist)

    # Map distance to confidence: 0mm -> 0.0, 4mm+ -> 1.0
    # Using sigmoid-like curve for smooth transition
    max_safe_dist = 4.0  # mm, distance at which we're fully confident
    if min_dist <= WIRE_HALF_WIDTH_MM:
        return 0.0
    effective_dist = min_dist - WIRE_HALF_WIDTH_MM
    return min(1.0, effective_dist / (max_safe_dist - WIRE_HALF_WIDTH_MM))


def _min_ring_distance(radius_mm: float) -> float:
    """Minimum distance from radius to any ring boundary."""
    if not RING_BOUNDARIES:
        return float("inf")
    return min(abs(radius_mm - r) for r in RING_BOUNDARIES)


def _min_sector_distance(radius_mm: float, angle_deg: float) -> float:
    """Minimum distance to nearest sector wire.

    Sector wires are radial lines at every 18 degrees (offset by 9 degrees
    so sector 20 is centered at 0 degrees). Only relevant in the scoring
    area (between outer_bull and double_outer).
    """
    # Bull area has no sector wires
    if radius_mm <= 15.9:
        return float("inf")
    # Outside board
    if radius_mm > 170.0:
        return float("inf")

    # Sector boundaries at 9, 27, 45, ... degrees
    half_sector = SECTOR_WIDTH_DEG / 2  # 9 degrees
    # Distance to nearest sector boundary in degrees
    shifted = (angle_deg + half_sector) % SECTOR_WIDTH_DEG
    angle_to_wire = min(shifted, SECTOR_WIDTH_DEG - shifted)

    # Convert angular distance to mm at this radius
    arc_dist_mm = radius_mm * math.radians(angle_to_wire)
    return arc_dist_mm


def detection_quality_confidence(
    contour_area: float | None = None,
    contour_solidity: float | None = None,
) -> float:
    """Confidence based on detection quality metrics.

    Args:
        contour_area: Detected contour area in pixels. Too small or too large
            indicates poor detection.
        contour_solidity: Ratio of contour area to convex hull area (0-1).
            Higher is better (dart-shaped contours are fairly solid).

    Returns:
        0.0 to 1.0 confidence.
    """
    if contour_area is None and contour_solidity is None:
        return 0.8  # Default when no detection metrics available

    scores = []

    if contour_area is not None:
        if contour_area < MIN_GOOD_CONTOUR_AREA:
            # Too small — likely noise
            scores.append(max(0.0, contour_area / MIN_GOOD_CONTOUR_AREA))
        elif contour_area > MAX_GOOD_CONTOUR_AREA:
            # Too large — likely multiple objects or bad segmentation
            excess = (contour_area - MAX_GOOD_CONTOUR_AREA) / MAX_GOOD_CONTOUR_AREA
            scores.append(max(0.0, 1.0 - excess))
        else:
            scores.append(1.0)

    if contour_solidity is not None:
        # Good darts have solidity > 0.5
        scores.append(min(1.0, max(0.0, contour_solidity / 0.7)))

    return sum(scores) / len(scores) if scores else 0.8


def triangulation_confidence(reprojection_error: float | None = None) -> float:
    """Confidence based on stereo triangulation quality.

    Args:
        reprojection_error: Reprojection error in pixels. Lower is better.

    Returns:
        0.0 to 1.0 confidence. Returns 0.8 default if no stereo data.
    """
    if reprojection_error is None:
        return 0.8  # Default for single-camera mode

    # 0 px error -> 1.0, 5+ px error -> 0.0
    max_error = 5.0
    if reprojection_error <= 0:
        return 1.0
    return max(0.0, 1.0 - reprojection_error / max_error)


def needs_review(confidence: float) -> bool:
    """Check if a hit needs manual review."""
    return confidence < REVIEW_THRESHOLD
