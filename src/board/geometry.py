"""
Dartboard geometry calculations and sector mapping.
"""
import numpy as np
from typing import Tuple, Optional
import logging

from src.core import BoardGeometry, Hit

logger = logging.getLogger(__name__)


class DartboardMapper:
    """
    Maps pixel coordinates to dartboard sectors and scores.

    Uses official dartboard dimensions and sector sequence to
    calculate scores from calibrated pixel coordinates.
    """

    # Official sector sequence (clockwise from top)
    SECTOR_SEQUENCE = (20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                       3, 19, 7, 16, 8, 11, 14, 9, 12, 5)

    def __init__(
            self,
            board_geometry: Optional[BoardGeometry] = None,
            board_center: Tuple[float, float] = (0, 0),
            mm_per_pixel: float = 1.0
    ):
        """
        Initialize dartboard mapper.

        Args:
            board_geometry: Board dimensions (default: BoardGeometry())
            board_center: Board center in pixel coordinates (x, y)
            mm_per_pixel: Scaling factor from calibration
        """
        self.geometry = board_geometry or BoardGeometry()
        self.center = board_center
        self.mm_per_pixel = mm_per_pixel

        # Convert mm radii to pixels
        self.inner_bull_radius_px = self.geometry.inner_bull_radius / mm_per_pixel
        self.outer_bull_radius_px = self.geometry.outer_bull_radius / mm_per_pixel
        self.triple_inner_radius_px = self.geometry.triple_inner_radius / mm_per_pixel
        self.triple_outer_radius_px = self.geometry.triple_outer_radius / mm_per_pixel
        self.double_inner_radius_px = self.geometry.double_inner_radius / mm_per_pixel
        self.double_outer_radius_px = self.geometry.double_outer_radius / mm_per_pixel

        logger.info(
            f"DartboardMapper initialized: center={board_center}, "
            f"scale={mm_per_pixel:.4f} mm/px"
        )

    def pixel_to_polar(
            self,
            x: float,
            y: float
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to polar coordinates.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels

        Returns:
            (radius, angle) where:
                - radius: Distance from center in pixels
                - angle: Angle in degrees (0° = top, clockwise)
        """
        # Calculate relative position
        dx = x - self.center[0]
        dy = y - self.center[1]

        # Radius
        radius = np.sqrt(dx ** 2 + dy ** 2)

        # Angle (convert from standard math angle to dartboard angle)
        # Math: 0° = right, counter-clockwise
        # Dartboard: 0° = top, clockwise
        angle_rad = np.arctan2(dy, dx)  # -π to π
        angle_deg = np.degrees(angle_rad)  # -180 to 180

        # Convert to dartboard convention:
        # 1. Rotate by 90° (top becomes 0°)
        # 2. Flip direction (clockwise positive)
        dartboard_angle = 90 - angle_deg

        # Normalize to [0, 360)
        if dartboard_angle < 0:
            dartboard_angle += 360

        return radius, dartboard_angle

    def angle_to_sector(self, angle: float) -> int:
        """
        Convert angle to sector number.

        Args:
            angle: Angle in degrees (0° = top, clockwise)

        Returns:
            Sector number (1-20)
        """
        # Each sector is 18° wide
        # Sector 20 is centered at 0° (top), spanning [-9°, 9°)

        # Adjust angle to center first sector at 0°
        adjusted_angle = (angle + self.geometry.sector_angle / 2) % 360

        # Calculate sector index
        sector_idx = int(adjusted_angle / self.geometry.sector_angle)

        # Map to sector number
        return self.SECTOR_SEQUENCE[sector_idx]

    def radius_to_ring(self, radius: float) -> Tuple[str, int]:
        """
        Convert radius to ring type and multiplier.

        Args:
            radius: Distance from center in pixels

        Returns:
            (ring_name, multiplier) where:
                - ring_name: "double_bull", "single_bull", "triple", "double", "single", "miss"
                - multiplier: 50, 25, 3, 2, 1, 0
        """
        if radius <= self.inner_bull_radius_px:
            return "double_bull", 50

        elif radius <= self.outer_bull_radius_px:
            return "single_bull", 25

        elif self.triple_inner_radius_px <= radius <= self.triple_outer_radius_px:
            return "triple", 3

        elif self.double_inner_radius_px <= radius <= self.double_outer_radius_px:
            return "double", 2

        elif radius < self.triple_inner_radius_px:
            # Between outer bull and triple (inner single area)
            return "single", 1

        elif radius < self.double_inner_radius_px:
            # Between triple and double (outer single area)
            return "single", 1

        else:
            # Outside board
            return "miss", 0

    def pixel_to_score(
            self,
            x: float,
            y: float,
            frame_id: Optional[int] = None,
            timestamp: Optional[float] = None
    ) -> Hit:
        """
        Convert pixel coordinates to score.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            frame_id: Optional frame ID for tracking
            timestamp: Optional timestamp

        Returns:
            Hit object with sector, multiplier, and score
        """
        # Convert to polar
        radius, angle = self.pixel_to_polar(x, y)

        # Determine ring and multiplier
        ring_name, multiplier = self.radius_to_ring(radius)

        # Determine sector (only for non-bull hits)
        if multiplier in [50, 25]:
            sector = None
            score = multiplier
        elif multiplier == 0:
            sector = None
            score = 0
        else:
            sector = self.angle_to_sector(angle)
            score = sector * multiplier

        # Create Hit object
        hit = Hit(
            x_px=x,
            y_px=y,
            radius=radius,
            angle=angle,
            sector=sector,
            multiplier=multiplier,
            score=score,
            frame_id=frame_id,
            timestamp=timestamp
        )

        logger.debug(
            f"Score: ({x:.1f}, {y:.1f}) → r={radius:.1f}px, θ={angle:.1f}° → "
            f"{ring_name} {sector if sector else ''} = {score}"
        )

        return hit

    def is_valid_hit(self, x: float, y: float) -> bool:
        """
        Check if coordinates are within board boundaries.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels

        Returns:
            True if within board, False otherwise
        """
        radius, _ = self.pixel_to_polar(x, y)
        return radius <= self.double_outer_radius_px

    def get_ring_boundaries(self) -> dict:
        """
        Get all ring boundaries in pixels.

        Returns:
            Dictionary with ring names and radii
        """
        return {
            "inner_bull": self.inner_bull_radius_px,
            "outer_bull": self.outer_bull_radius_px,
            "triple_inner": self.triple_inner_radius_px,
            "triple_outer": self.triple_outer_radius_px,
            "double_inner": self.double_inner_radius_px,
            "double_outer": self.double_outer_radius_px,
        }

    def get_sector_boundaries(self) -> list:
        """
        Get sector boundary angles.

        Returns:
            List of (sector_number, start_angle, end_angle) tuples
        """
        boundaries = []
        half_sector = self.geometry.sector_angle / 2

        for i, sector_num in enumerate(self.SECTOR_SEQUENCE):
            # Center angle for this sector
            center_angle = i * self.geometry.sector_angle

            # Boundaries
            start_angle = (center_angle - half_sector) % 360
            end_angle = (center_angle + half_sector) % 360

            boundaries.append((sector_num, start_angle, end_angle))

        return boundaries