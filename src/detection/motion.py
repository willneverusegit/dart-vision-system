"""
Motion detection using MOG2 background subtraction.
Implements Motion Gating for CPU budget optimization.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotionConfig:
    """Configuration for motion detection."""
    # MOG2 parameters
    history: int = 500  # Number of frames for background model
    var_threshold: float = 16.0  # Detection threshold (lower = more sensitive)
    detect_shadows: bool = False  # Shadow detection (slower, usually not needed)
    learning_rate: float = -1.0  # Automatic learning rate

    # Motion thresholds
    min_motion_area: int = 50  # Minimum pixels to consider as motion
    motion_threshold: int = 127  # Binary threshold for motion mask

    # Morphological operations (noise reduction)
    enable_morphology: bool = True
    morph_kernel_size: int = 3
    morph_iterations: int = 1

    # ← NEU: ROI support
    roi_margin_px: int = 50  # Margin around board for motion detection

class MotionDetector:
    """
    Detects motion using MOG2 background subtraction.

    Optimized for CPU performance:
    - MOG2: 15-25 FPS on low-end laptops
    - Motion gating: Only trigger processing when motion detected
    - Morphological ops: Reduce noise

    Example:
        detector = MotionDetector()

        for frame in video_stream:
            motion_mask, has_motion = detector.detect(frame.image)

            if has_motion:
                # Process frame (expensive operations)
                analyze_for_darts(frame)
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        """
        Initialize motion detector.

        Args:
            config: Motion detection configuration
        """
        self.config = config or MotionConfig()

        # Initialize MOG2 background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows
        )

        # Morphological kernel for noise reduction
        if self.config.enable_morphology:
            self.morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.morph_kernel_size, self.config.morph_kernel_size)
            )
        else:
            self.morph_kernel = None

        # Statistics
        self.frame_count = 0
        self.motion_detected_count = 0

        logger.info(
            f"MotionDetector initialized: history={self.config.history}, "
            f"threshold={self.config.var_threshold}"
        )

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detect motion in image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            (motion_mask, has_motion) where:
                - motion_mask: Binary mask (uint8, 0 or 255)
                - has_motion: True if significant motion detected
        """
        self.frame_count += 1

        # Convert to grayscale if needed (MOG2 works with grayscale or BGR)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(
            gray,
            learningRate=self.config.learning_rate
        )

        # Threshold to binary
        _, motion_mask = cv2.threshold(
            fg_mask,
            self.config.motion_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Apply morphological operations (reduce noise)
        if self.config.enable_morphology and self.morph_kernel is not None:
            # Opening: erosion followed by dilation (removes small noise)
            motion_mask = cv2.morphologyEx(
                motion_mask,
                cv2.MORPH_OPEN,
                self.morph_kernel,
                iterations=self.config.morph_iterations
            )

            # Closing: dilation followed by erosion (fills small holes)
            motion_mask = cv2.morphologyEx(
                motion_mask,
                cv2.MORPH_CLOSE,
                self.morph_kernel,
                iterations=self.config.morph_iterations
            )

        # Check if significant motion present
        motion_pixels = cv2.countNonZero(motion_mask)
        has_motion = motion_pixels >= self.config.min_motion_area

        if has_motion:
            self.motion_detected_count += 1
            logger.debug(f"Motion detected: {motion_pixels} pixels")

        return motion_mask, has_motion

    def reset(self) -> None:
        """Reset background model (useful when camera moves)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows
        )
        logger.info("Background model reset")

    @property
    def motion_rate(self) -> float:
        """Get percentage of frames with motion detected."""
        if self.frame_count == 0:
            return 0.0
        return (self.motion_detected_count / self.frame_count) * 100.0

    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            "frames_processed": self.frame_count,
            "motion_detected": self.motion_detected_count,
            "motion_rate_percent": self.motion_rate,
        }