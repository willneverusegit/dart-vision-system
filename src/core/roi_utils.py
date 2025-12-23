"""
ROI (Region of Interest) utilities for efficient processing.
"""
import numpy as np
from typing import Tuple, Optional
import cv2
import logging

from .types import ROI, CalibrationData

logger = logging.getLogger(__name__)


class ROIExtractor:
    """
    Extracts and manages Region of Interest for efficient processing.

    ROI-First strategy reduces processing by 80-90% by cropping to
    only the dartboard area.
    """

    def __init__(
            self,
            calibration: CalibrationData,
            margin_px: int = 50,
            target_size: int = 800
    ):
        """
        Initialize ROI extractor.

        Args:
            calibration: Calibration data with board center and radius
            margin_px: Extra margin around board (default: 50px)
            target_size: Target size for warped output (default: 800x800)
        """
        self.calibration = calibration
        self.margin_px = margin_px
        self.target_size = target_size

        # Calculate ROI bounds from board geometry
        self.roi = self._calculate_roi()

        logger.info(
            f"ROIExtractor initialized: "
            f"ROI={self.roi.width}x{self.roi.height}, "
            f"margin={margin_px}px"
        )

    def _calculate_roi(self) -> ROI:
        """
        Calculate ROI from calibration data.

        Returns:
            ROI object with bounding box
        """
        # Get board bounds with margin
        center_x, center_y = self.calibration.board_center
        radius = self.calibration.board_radius_px

        # Add margin
        x_min = int(max(0, center_x - radius - self.margin_px))
        y_min = int(max(0, center_y - radius - self.margin_px))
        x_max = int(center_x + radius + self.margin_px)
        y_max = int(center_y + radius + self.margin_px)

        width = x_max - x_min
        height = y_max - y_min

        return ROI(x=x_min, y=y_min, width=width, height=height)

    def extract_roi(self, image: np.ndarray) -> Tuple[np.ndarray, ROI]:
        """
        Extract ROI from image.

        Args:
            image: Full-size image

        Returns:
            (cropped_image, roi) tuple
        """
        h, w = image.shape[:2]

        # Clamp ROI to image bounds
        x1 = max(0, self.roi.x)
        y1 = max(0, self.roi.y)
        x2 = min(w, self.roi.x + self.roi.width)
        y2 = min(h, self.roi.y + self.roi.height)

        cropped = image[y1:y2, x1:x2]
        actual_roi = ROI(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

        return cropped, actual_roi

    def roi_to_full_coords(
            self,
            x_roi: float,
            y_roi: float
    ) -> Tuple[float, float]:
        """
        Convert ROI coordinates to full image coordinates.

        Args:
            x_roi: X coordinate in ROI
            y_roi: Y coordinate in ROI

        Returns:
            (x_full, y_full) in full image coordinates
        """
        x_full = x_roi + self.roi.x
        y_full = y_roi + self.roi.y
        return x_full, y_full

    def full_to_roi_coords(
            self,
            x_full: float,
            y_full: float
    ) -> Tuple[float, float]:
        """
        Convert full image coordinates to ROI coordinates.

        Args:
            x_full: X coordinate in full image
            y_full: Y coordinate in full image

        Returns:
            (x_roi, y_roi) in ROI coordinates
        """
        x_roi = x_full - self.roi.x
        y_roi = y_full - self.roi.y
        return x_roi, y_roi


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for frames.

    Applies operations in optimal order for performance:
    1. ROI cropping (80-90% data reduction)
    2. Downscaling (further reduction)
    3. Grayscale conversion (3x speedup)
    4. Optional CLAHE (contrast enhancement)
    """

    def __init__(
            self,
            roi_extractor: Optional[ROIExtractor] = None,
            target_width: Optional[int] = None,
            enable_grayscale: bool = True,
            enable_clahe: bool = False,
            clahe_clip_limit: float = 2.0,
            clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            roi_extractor: ROI extractor (None = no ROI cropping)
            target_width: Target width for downscaling (None = no downscaling)
            enable_grayscale: Convert to grayscale
            enable_clahe: Apply CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_grid_size: CLAHE tile grid size
        """
        self.roi_extractor = roi_extractor
        self.target_width = target_width
        self.enable_grayscale = enable_grayscale
        self.enable_clahe = enable_clahe

        # CLAHE
        if enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_tile_grid_size
            )
        else:
            self.clahe = None

        logger.info(
            f"PreprocessingPipeline: "
            f"ROI={'Yes' if roi_extractor else 'No'}, "
            f"Downscale={target_width if target_width else 'No'}, "
            f"Grayscale={enable_grayscale}, "
            f"CLAHE={enable_clahe}"
        )

    def process(
            self,
            image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[ROI]]:
        """
        Process image through pipeline.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            (processed_image, roi) where roi is None if not used
        """
        processed = image
        roi = None

        # Step 1: ROI cropping (biggest win)
        if self.roi_extractor:
            processed, roi = self.roi_extractor.extract_roi(processed)

        # Step 2: Downscaling
        if self.target_width and processed.shape[1] > self.target_width:
            h, w = processed.shape[:2]
            aspect_ratio = h / w
            target_height = int(self.target_width * aspect_ratio)

            processed = cv2.resize(
                processed,
                (self.target_width, target_height),
                interpolation=cv2.INTER_AREA
            )

        # Step 3: Grayscale conversion
        if self.enable_grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Step 4: CLAHE (optional)
        if self.enable_clahe and self.clahe:
            if len(processed.shape) == 3:
                # Convert to grayscale first
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = self.clahe.apply(processed)

        return processed, roi