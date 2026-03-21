"""Background subtraction and motion detection for dart hit detection."""

import logging
import time

import cv2
import numpy as np

from backend.vision.camera import camera_manager

logger = logging.getLogger(__name__)


class BackgroundModel:
    """Stores and manages the reference frame (empty board)."""

    def __init__(self) -> None:
        self._background: np.ndarray | None = None

    def capture_background(self, camera_id: int) -> bool:
        """Grab a frame from the camera and store as grayscale background.

        Returns True on success.
        """
        frame = camera_manager.capture_frame(camera_id)
        if frame is None:
            logger.error("Failed to capture background from camera %d", camera_id)
            return False
        self._background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logger.info("Background captured from camera %d", camera_id)
        return True

    def set_background(self, frame: np.ndarray) -> None:
        """Set background directly from a frame (BGR or grayscale)."""
        if len(frame.shape) == 3:
            self._background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self._background = frame.copy()

    def get_background(self) -> np.ndarray | None:
        """Return the stored grayscale background frame."""
        return self._background


def detect_motion(
    frame: np.ndarray,
    background: np.ndarray,
    threshold: int = 25,
    min_area: int = 500,
) -> bool:
    """Detect motion by comparing a frame against the background.

    Args:
        frame: Current frame (BGR or grayscale).
        background: Reference background (grayscale).
        threshold: Pixel intensity difference threshold.
        min_area: Minimum contour area to count as motion.

    Returns:
        True if significant motion is detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    diff = cv2.absdiff(gray, background)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            return True
    return False


def wait_for_stable_frame(
    camera_id: int,
    delay_ms: int = 300,
    max_retries: int = 3,
    stability_threshold: int = 10,
) -> np.ndarray | None:
    """Wait until consecutive frames stabilize after motion is detected.

    Args:
        camera_id: Camera index.
        delay_ms: Initial delay before first capture (vibration settle).
        max_retries: Maximum attempts to get a stable frame.
        stability_threshold: Max mean diff between consecutive frames to count as stable.

    Returns:
        Stable BGR frame or None if stabilization fails.
    """
    time.sleep(delay_ms / 1000.0)

    prev_frame = camera_manager.capture_frame(camera_id)
    if prev_frame is None:
        return None

    for _ in range(max_retries):
        time.sleep(0.05)  # 50ms between checks
        curr_frame = camera_manager.capture_frame(camera_id)
        if curr_frame is None:
            return None

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = float(np.mean(diff))

        if mean_diff < stability_threshold:
            logger.debug("Stable frame acquired (mean_diff=%.1f)", mean_diff)
            return curr_frame

        prev_frame = curr_frame

    logger.warning("Frame stabilization failed after %d retries", max_retries)
    return prev_frame  # Return last frame anyway
