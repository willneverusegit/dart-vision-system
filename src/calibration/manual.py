"""
Manual calibration using 4-point selection.
Simple and fast fallback method.
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
import time
import logging

from src.core import CalibrationData, BoardGeometry, Frame
from .base import CalibratorBase, CalibrationResult

logger = logging.getLogger(__name__)


class ManualCalibrator(CalibratorBase):
    """
    Manual calibration by clicking 4 board points.

    User clicks 4 points on the outer double ring:
    - Top (20 sector at 12 o'clock)
    - Right (6 sector at 3 o'clock)
    - Bottom (3 sector at 6 o'clock)
    - Left (11 sector at 9 o'clock)

    Accuracy: ~2-5 mm (depends on user precision)
    """

    def __init__(
            self,
            board_geometry: Optional[BoardGeometry] = None,
            canonical_size: int = 800
    ):
        """
        Initialize manual calibrator.

        Args:
            board_geometry: Dartboard geometry
            canonical_size: Size of canonical (square) output in pixels
        """
        super().__init__(board_geometry)
        self.canonical_size = canonical_size

        # For interactive selection
        self._selected_points: List[Tuple[int, int]] = []
        self._display_image: Optional[np.ndarray] = None
        self._window_name = "Manual Calibration"

    def calibrate(self, frame: Frame) -> CalibrationResult:
        """
        Perform manual calibration on a frame.

        Args:
            frame: Input frame to calibrate on

        Returns:
            CalibrationResult with success status and data
        """
        logger.info("Starting manual calibration...")

        # Interactive point selection
        self._display_image = frame.image.copy()
        self._selected_points = []

        # Setup window and mouse callback
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

        # Instructions
        self._draw_instructions()

        print("\n" + "=" * 60)
        print("Manual Calibration")
        print("=" * 60)
        print("Click 4 points on the OUTER DOUBLE RING:")
        print("  1. Top    (20 sector, 12 o'clock)")
        print("  2. Right  ( 6 sector,  3 o'clock)")
        print("  3. Bottom ( 3 sector,  6 o'clock)")
        print("  4. Left   (11 sector,  9 o'clock)")
        print("\nPress 'r' to reset, 'q' to cancel")
        print("=" * 60 + "\n")

        # Wait for 4 points
        while len(self._selected_points) < 4:
            cv2.imshow(self._window_name, self._display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyWindow(self._window_name)
                return CalibrationResult(
                    success=False,
                    data=None,
                    message="Calibration cancelled by user"
                )
            elif key == ord('r'):
                self._reset_selection()

        cv2.destroyWindow(self._window_name)

        # Compute calibration
        return self._compute_calibration(frame.image)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self._selected_points) < 4:
            self._selected_points.append((x, y))
            logger.debug(f"Point {len(self._selected_points)} selected: ({x}, {y})")

            # Redraw
            self._display_image = self._display_image.copy()
            self._draw_points()
            self._draw_instructions()

    def _draw_points(self):
        """Draw selected points on display image."""
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
        labels = ["1: Top", "2: Right", "3: Bottom", "4: Left"]

        for i, (x, y) in enumerate(self._selected_points):
            # Draw point
            cv2.circle(self._display_image, (x, y), 8, colors[i], -1)
            cv2.circle(self._display_image, (x, y), 10, (255, 255, 255), 2)

            # Draw label
            cv2.putText(
                self._display_image,
                labels[i],
                (x + 15, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colors[i],
                2
            )

        # Draw lines between points
        if len(self._selected_points) > 1:
            for i in range(len(self._selected_points) - 1):
                cv2.line(
                    self._display_image,
                    self._selected_points[i],
                    self._selected_points[i + 1],
                    (0, 255, 255),
                    2
                )

    def _draw_instructions(self):
        """Draw instruction overlay."""
        text = f"Select point {len(self._selected_points) + 1}/4"
        if len(self._selected_points) < 4:
            point_names = ["TOP (12 o'clock)", "RIGHT (3 o'clock)",
                           "BOTTOM (6 o'clock)", "LEFT (9 o'clock)"]
            text += f" - {point_names[len(self._selected_points)]}"
        else:
            text = "All points selected. Processing..."

        # Background rectangle
        cv2.rectangle(
            self._display_image,
            (10, 10),
            (600, 50),
            (0, 0, 0),
            -1
        )

        # Text
        cv2.putText(
            self._display_image,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    def _reset_selection(self):
        """Reset point selection."""
        self._selected_points = []
        self._display_image = self._display_image.copy()
        self._draw_instructions()
        logger.info("Selection reset")

    def _compute_calibration(self, image: np.ndarray) -> CalibrationResult:
        """Compute calibration from selected points."""
        logger.info("Computing calibration from 4 points...")

        # Convert to numpy array
        src_points = np.array(self._selected_points, dtype=np.float32)

        # Create canonical destination points (square board)
        # Place points at edges of canonical square
        half_size = self.canonical_size / 2
        dst_points = np.array([
            [half_size, 0],  # Top
            [self.canonical_size, half_size],  # Right
            [half_size, self.canonical_size],  # Bottom
            [0, half_size]  # Left
        ], dtype=np.float32)

        # Compute homography
        success, H = self.compute_homography(src_points, dst_points)

        if not success or H is None:
            return CalibrationResult(
                success=False,
                data=None,
                message="Failed to compute homography"
            )

        # Warp image to verify
        warped = self.warp_image(image, H, (self.canonical_size, self.canonical_size))

        # Calculate board center (should be at center of canonical view)
        board_center = (half_size, half_size)

        # Calculate board radius (distance from center to any edge point)
        board_radius_px = half_size

        # Calculate mm_per_pixel
        # Official dartboard outer double ring diameter: 340 mm (radius = 170 mm)
        known_radius_mm = self.board_geometry.double_outer_radius  # 170 mm
        mm_per_px = self.calculate_mm_per_pixel(known_radius_mm, board_radius_px)

        # Create calibration data
        calib_data = CalibrationData(
            homography_matrix=H,
            board_center=board_center,
            mm_per_pixel=mm_per_px,
            board_radius_px=board_radius_px,
            method="manual",
            timestamp=time.time()
        )

        # Validate
        is_valid, msg = self.validate_calibration(calib_data)

        if not is_valid:
            return CalibrationResult(
                success=False,
                data=None,
                message=f"Calibration validation failed: {msg}"
            )

        logger.info(f"Manual calibration successful: {mm_per_px:.4f} mm/px")

        return CalibrationResult(
            success=True,
            data=calib_data,
            message="Manual calibration completed successfully",
            warped_image=warped
        )