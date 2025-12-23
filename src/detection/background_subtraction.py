"""
Background subtraction with multiple algorithms.
Supports MOG2, KNN for different scenarios.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BackgroundMethod(Enum):
    """Background subtraction methods."""
    MOG2 = "mog2"
    KNN = "knn"


class AdaptiveBackgroundSubtractor:
    """
    Adaptive background subtraction with multiple algorithms.

    Based on research:
    - MOG2: Good all-round performance, adaptive
    - KNN: Better for scenes with little foreground movement

    References:
    - KNN showed better segmentation in challenging conditions
    - MOG2 with history=100-200 recommended for CPU
    """

    def __init__(
            self,
            method: BackgroundMethod = BackgroundMethod.MOG2,
            history: int = 150,  # Reduced from 500 (research recommendation)
            var_threshold: float = 25.0,  # Mid-range 16-32
            detect_shadows: bool = False,
            dist2_threshold: float = 400.0,  # For KNN
            enable_morphology: bool = True,
            morph_kernel_size: int = 3,
            morph_iterations: int = 1
    ):
        """
        Initialize background subtractor.

        Args:
            method: Background subtraction method
            history: Number of frames for background model (100-200 recommended)
            var_threshold: Detection threshold (16-32 recommended)
            detect_shadows: Detect shadows (expensive, usually false)
            dist2_threshold: KNN distance threshold
            enable_morphology: Apply morphological operations
            morph_kernel_size: Kernel size for morphology
            morph_iterations: Number of morphology iterations
        """
        self.method = method
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.dist2_threshold = dist2_threshold

        # Create background subtractor
        if method == BackgroundMethod.MOG2:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )
            logger.info(f"Using MOG2: history={history}, varThreshold={var_threshold}")

        elif method == BackgroundMethod.KNN:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=dist2_threshold,
                detectShadows=detect_shadows
            )
            logger.info(f"Using KNN: history={history}, dist2Threshold={dist2_threshold}")

        # Morphological operations
        self.enable_morphology = enable_morphology
        if enable_morphology:
            self.morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (morph_kernel_size, morph_kernel_size)
            )
            self.morph_iterations = morph_iterations
        else:
            self.morph_kernel = None
            self.morph_iterations = 0

        # Statistics
        self.frame_count = 0

    def apply(
            self,
            image: np.ndarray,
            learning_rate: float = -1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply background subtraction.

        Args:
            image: Input image (BGR or grayscale)
            learning_rate: Learning rate (-1 = automatic)

        Returns:
            (fg_mask, cleaned_mask) tuple
        """
        self.frame_count += 1

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray, learningRate=learning_rate)

        # Threshold to binary
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Apply morphological operations
        cleaned_mask = fg_mask
        if self.enable_morphology and self.morph_kernel is not None:
            # Opening: removes small noise
            cleaned_mask = cv2.morphologyEx(
                cleaned_mask,
                cv2.MORPH_OPEN,
                self.morph_kernel,
                iterations=self.morph_iterations
            )

            # Closing: fills small holes
            cleaned_mask = cv2.morphologyEx(
                cleaned_mask,
                cv2.MORPH_CLOSE,
                self.morph_kernel,
                iterations=self.morph_iterations
            )

        return fg_mask, cleaned_mask

    def reset(self) -> None:
        """Reset background model."""
        if self.method == BackgroundMethod.MOG2:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows
            )
        elif self.method == BackgroundMethod.KNN:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.dist2_threshold,
                detectShadows=self.detect_shadows
            )

        logger.info(f"Background model reset ({self.method.value})")