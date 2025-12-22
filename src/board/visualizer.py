"""
Board visualization and overlay rendering.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

from src.core import Hit
from .geometry import DartboardMapper

logger = logging.getLogger(__name__)


class BoardVisualizer:
    """
    Renders dartboard overlays and hit markers.

    Provides visual debugging for:
    - Board geometry (sectors, rings)
    - Hit locations
    - Scores
    """

    # Color scheme
    COLORS = {
        "center": (0, 0, 255),  # Red
        "rings": (0, 255, 0),  # Green
        "sectors": (255, 255, 0),  # Cyan
        "hit": (0, 0, 255),  # Red
        "text": (255, 255, 255),  # White
        "overlay": (0, 255, 255),  # Yellow
    }

    def __init__(
            self,
            mapper: DartboardMapper,
            opacity: float = 0.3
    ):
        """
        Initialize visualizer.

        Args:
            mapper: DartboardMapper for geometry
            opacity: Overlay opacity (0.0-1.0)
        """
        self.mapper = mapper
        self.opacity = max(0.0, min(1.0, opacity))

    def draw_board_overlay(
            self,
            image: np.ndarray,
            draw_rings: bool = True,
            draw_sectors: bool = True,
            draw_center: bool = True
    ) -> np.ndarray:
        """
        Draw board geometry overlay on image.

        Args:
            image: Input image (will be modified)
            draw_rings: Draw ring boundaries
            draw_sectors: Draw sector boundaries
            draw_center: Draw center crosshair

        Returns:
            Image with overlay
        """
        overlay = image.copy()
        center = tuple(map(int, self.mapper.center))

        # Draw rings
        if draw_rings:
            boundaries = self.mapper.get_ring_boundaries()

            for ring_name, radius in boundaries.items():
                cv2.circle(
                    overlay,
                    center,
                    int(radius),
                    self.COLORS["rings"],
                    2
                )

        # Draw sectors
        if draw_sectors:
            sector_boundaries = self.mapper.get_sector_boundaries()
            outer_radius = int(self.mapper.double_outer_radius_px)

            for sector_num, start_angle, end_angle in sector_boundaries:
                # Draw sector line (from center to outer edge)
                angle_rad = np.radians(start_angle)
                end_x = int(center[0] + outer_radius * np.sin(angle_rad))
                end_y = int(center[1] - outer_radius * np.cos(angle_rad))

                cv2.line(
                    overlay,
                    center,
                    (end_x, end_y),
                    self.COLORS["sectors"],
                    1
                )

                # Draw sector number (at mid-angle, outer single ring)
                mid_angle = (start_angle + self.mapper.geometry.sector_angle / 2) % 360
                text_radius = (self.mapper.triple_outer_radius_px +
                               self.mapper.double_inner_radius_px) / 2

                text_angle_rad = np.radians(mid_angle)
                text_x = int(center[0] + text_radius * np.sin(text_angle_rad))
                text_y = int(center[1] - text_radius * np.cos(text_angle_rad))

                cv2.putText(
                    overlay,
                    str(sector_num),
                    (text_x - 10, text_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLORS["text"],
                    1
                )

        # Draw center
        if draw_center:
            cv2.circle(overlay, center, 5, self.COLORS["center"], -1)
            cv2.circle(overlay, center, 8, self.COLORS["center"], 2)

            # Crosshair
            crosshair_len = 20
            cv2.line(
                overlay,
                (center[0] - crosshair_len, center[1]),
                (center[0] + crosshair_len, center[1]),
                self.COLORS["center"],
                2
            )
            cv2.line(
                overlay,
                (center[0], center[1] - crosshair_len),
                (center[0], center[1] + crosshair_len),
                self.COLORS["center"],
                2
            )

        # Blend with original
        result = cv2.addWeighted(image, 1 - self.opacity, overlay, self.opacity, 0)

        return result

    def draw_hit(
            self,
            image: np.ndarray,
            hit: Hit,
            show_score: bool = True,
            show_coords: bool = False
    ) -> np.ndarray:
        """
        Draw hit marker on image.

        Args:
            image: Input image (will be modified)
            hit: Hit object
            show_score: Show score text
            show_coords: Show pixel coordinates

        Returns:
            Image with hit marker
        """
        result = image.copy()

        pos = (int(hit.x_px), int(hit.y_px))

        # Draw hit marker
        cv2.circle(result, pos, 8, self.COLORS["hit"], -1)
        cv2.circle(result, pos, 12, (255, 255, 255), 2)

        # Build score text
        if show_score and hit.score is not None:
            if hit.multiplier == 50:
                score_text = "BULL (50)"
            elif hit.multiplier == 25:
                score_text = "25"
            elif hit.multiplier == 0:
                score_text = "MISS"
            else:
                multiplier_text = {1: "", 2: "D", 3: "T"}[hit.multiplier]
                score_text = f"{multiplier_text}{hit.sector} ({hit.score})"

            # Draw score text
            text_offset_y = -20
            cv2.putText(
                result,
                score_text,
                (pos[0] + 15, pos[1] + text_offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.COLORS["text"],
                2
            )

        # Draw coordinates
        if show_coords:
            coord_text = f"({hit.x_px:.1f}, {hit.y_px:.1f})"
            cv2.putText(
                result,
                coord_text,
                (pos[0] + 15, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.COLORS["text"],
                1
            )

        return result

    def draw_hits(
            self,
            image: np.ndarray,
            hits: List[Hit],
            show_scores: bool = True
    ) -> np.ndarray:
        """
        Draw multiple hits on image.

        Args:
            image: Input image
            hits: List of Hit objects
            show_scores: Show score text for each hit

        Returns:
            Image with all hits drawn
        """
        result = image.copy()

        for hit in hits:
            result = self.draw_hit(result, hit, show_score=show_scores)

        return result

    def draw_info_panel(
            self,
            image: np.ndarray,
            info_dict: dict,
            position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Draw info panel with text overlay.

        Args:
            image: Input image
            info_dict: Dictionary with info to display
            position: Top-left position (x, y)

        Returns:
            Image with info panel
        """
        result = image.copy()

        x, y = position
        line_height = 25

        # Background rectangle
        max_text_width = max(len(f"{k}: {v}") for k, v in info_dict.items())
        rect_width = max_text_width * 10 + 20
        rect_height = len(info_dict) * line_height + 20

        cv2.rectangle(
            result,
            (x - 10, y - 20),
            (x + rect_width, y + rect_height),
            (0, 0, 0),
            -1
        )

        # Draw text
        for i, (key, value) in enumerate(info_dict.items()):
            text = f"{key}: {value}"
            cv2.putText(
                result,
                text,
                (x, y + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.COLORS["text"],
                2
            )

        return result