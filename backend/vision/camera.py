"""Camera capture module with device management and frame streaming."""

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraDevice:
    """Represents a connected camera device."""

    index: int
    name: str
    available: bool = True


class CameraManager:
    """Manages camera devices: open, capture, release."""

    def __init__(self) -> None:
        self._captures: dict[int, cv2.VideoCapture] = {}
        self._lock = threading.Lock()

    def list_devices(self, max_check: int = 5) -> list[CameraDevice]:
        """Probe for available camera devices."""
        devices = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append(CameraDevice(index=i, name=f"Camera {i}"))
                cap.release()
        return devices

    def open(self, index: int, width: int = 640, height: int = 480) -> bool:
        """Open a camera by index."""
        with self._lock:
            if index in self._captures:
                return True
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                logger.error("Failed to open camera %d", index)
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._captures[index] = cap
            logger.info("Opened camera %d at %dx%d", index, width, height)
            return True

    def close(self, index: int) -> None:
        """Release a camera by index."""
        with self._lock:
            cap = self._captures.pop(index, None)
            if cap:
                cap.release()
                logger.info("Closed camera %d", index)

    def close_all(self) -> None:
        """Release all cameras."""
        with self._lock:
            for cap in self._captures.values():
                cap.release()
            self._captures.clear()

    def capture_frame(self, index: int) -> np.ndarray | None:
        """Capture a single frame from a camera. Returns BGR numpy array or None."""
        with self._lock:
            cap = self._captures.get(index)
            if cap is None:
                return None
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera %d", index)
                return None
            return frame

    def capture_frame_jpeg(self, index: int, quality: int = 80) -> bytes | None:
        """Capture a frame and encode as JPEG bytes."""
        frame = self.capture_frame(index)
        if frame is None:
            return None
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return None
        return buffer.tobytes()

    def is_open(self, index: int) -> bool:
        """Check if a camera is currently open."""
        with self._lock:
            return index in self._captures

    def get_resolution(self, index: int) -> tuple[int, int] | None:
        """Get the current resolution of an open camera."""
        with self._lock:
            cap = self._captures.get(index)
            if cap is None:
                return None
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)


# Singleton instance
camera_manager = CameraManager()
