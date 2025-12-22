"""
Frame preprocessing for performance optimization.
Implements early data reduction strategies: ROI cropping, downscaling, CLAHE.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

from src.core import Frame, ROI

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for frame preprocessing."""
    # Downscaling
    enable_downscale: bool = True
    target_width: int = 1280

    # Color conversion
    convert_grayscale: bool = False  # 3x speedup but loses color

    # Contrast enhancement (for poor lighting)
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # ROI (set after calibration)
    roi: Optional[ROI] = None


class FramePreprocessor:
    """
    Efficient frame preprocessing with early data reduction.

    Processing order (optimized for performance):
    1. ROI cropping (if configured) - reduces data by 80-90%
    2. Downscaling (if enabled) - reduces resolution
    3. Grayscale conversion (if enabled) - 3x speedup
    4. CLAHE (if enabled) - contrast enhancement

    Example:
        config = PreprocessConfig(
            convert_grayscale=True,
            apply_clahe=True,
            roi=ROI(100, 100, 400, 400)
        )
        preprocessor = FramePreprocessor(config)

        processed = preprocessor.process(frame)
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessConfig()

        # Initialize CLAHE if needed
        self._clahe = None
        if self.config.apply_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size
            )
            logger.debug("CLAHE initialized")

    def process(self, frame: Frame) -> Frame:
        """
        Process frame with configured preprocessing steps.

        Args:
            frame: Input frame

        Returns:
            Processed frame (new Frame object with processed image)
        """
        image = frame.image.copy()

        # Step 1: ROI cropping (if configured)
        if self.config.roi is not None:
            image = self.config.roi.crop(image)

        # Step 2: Downscaling
        if self.config.enable_downscale:
            image = self._downscale(image)

        # Step 3: Grayscale conversion
        if self.config.convert_grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 4: CLAHE (contrast enhancement)
        if self.config.apply_clahe and self._clahe is not None:
            if len(image.shape) == 3:
                # Apply CLAHE to each channel
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale image
                image = self._clahe.apply(image)

        # Create new Frame with processed image
        return Frame(
            image=image,
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            fps=frame.fps
        )

    def _downscale(self, image: np.ndarray) -> np.ndarray:
        """
        Downscale image to target width (maintains aspect ratio).

        Args:
            image: Input image

        Returns:
            Downscaled image
        """
        height, width = image.shape[:2]

        # Skip if already smaller
        if width <= self.config.target_width:
            return image

        # Calculate new dimensions
        scale = self.config.target_width / width
        new_width = self.config.target_width
        new_height = int(height * scale)

        # Use INTER_AREA for downscaling (best quality)
        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    def set_roi(self, roi: ROI) -> None:
        """
        Set or update ROI for cropping.

        Args:
            roi: Region of interest
        """
        self.config.roi = roi
        logger.info(f"ROI updated: {roi}")

    def clear_roi(self) -> None:
        """Clear ROI (process full frame)."""
        self.config.roi = None
        logger.info("ROI cleared")

    def update_config(self, **kwargs) -> None:
        """
        Update preprocessing configuration.

        Example:
            preprocessor.update_config(
                convert_grayscale=True,
                apply_clahe=False
            )
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Config updated: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")