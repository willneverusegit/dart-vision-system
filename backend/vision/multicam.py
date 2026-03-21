"""Multi-camera management with frame synchronization."""

import logging
import threading
import time

import numpy as np

from backend.vision.camera import camera_manager

logger = logging.getLogger(__name__)


class MultiCameraManager:
    """Manages 2-3 cameras simultaneously with synchronized capture."""

    def __init__(self) -> None:
        self._camera_ids: list[int] = []
        self._lock = threading.Lock()

    def add_camera(self, index: int, width: int = 640, height: int = 480) -> bool:
        """Open a camera and add it to the managed set.

        Args:
            index: Camera device index.
            width: Desired frame width.
            height: Desired frame height.

        Returns:
            True if the camera was opened successfully.
        """
        with self._lock:
            if index in self._camera_ids:
                return True
            success = camera_manager.open(index, width, height)
            if success:
                self._camera_ids.append(index)
                logger.info("Added camera %d to multi-cam manager", index)
            return success

    def remove_camera(self, index: int) -> None:
        """Close a camera and remove it from the managed set.

        Args:
            index: Camera device index to remove.
        """
        with self._lock:
            if index in self._camera_ids:
                camera_manager.close(index)
                self._camera_ids.remove(index)
                logger.info("Removed camera %d from multi-cam manager", index)

    def get_camera_ids(self) -> list[int]:
        """Return list of currently managed camera indices."""
        with self._lock:
            return list(self._camera_ids)

    def capture_synchronized(
        self, max_time_diff_ms: float = 50.0
    ) -> dict[int, tuple[np.ndarray, float]]:
        """Capture frames from all managed cameras as close in time as possible.

        Args:
            max_time_diff_ms: Maximum acceptable time difference in milliseconds
                between first and last captured frame. A warning is logged if
                exceeded.

        Returns:
            Dict mapping camera index to (frame, timestamp) tuples.
            Cameras that fail to capture are omitted.
        """
        results: dict[int, tuple[np.ndarray, float]] = {}
        with self._lock:
            ids = list(self._camera_ids)

        for cam_id in ids:
            frame = camera_manager.capture_frame(cam_id)
            ts = time.time()
            if frame is not None:
                results[cam_id] = (frame, ts)
            else:
                logger.warning("Failed to capture frame from camera %d", cam_id)

        if len(results) >= 2:
            timestamps = [ts for _, ts in results.values()]
            diff_ms = (max(timestamps) - min(timestamps)) * 1000.0
            if diff_ms > max_time_diff_ms:
                logger.warning(
                    "Frame sync exceeded threshold: %.1f ms > %.1f ms",
                    diff_ms,
                    max_time_diff_ms,
                )

        return results

    def is_healthy(self, index: int) -> bool:
        """Check if a camera can capture a frame.

        Args:
            index: Camera device index.

        Returns:
            True if a frame was captured successfully.
        """
        frame = camera_manager.capture_frame(index)
        return frame is not None

    def health_check(self) -> dict[int, bool]:
        """Run health check on all managed cameras.

        Returns:
            Dict mapping camera index to health status.
        """
        with self._lock:
            ids = list(self._camera_ids)
        return {cam_id: self.is_healthy(cam_id) for cam_id in ids}

    def close_all(self) -> None:
        """Close and remove all managed cameras."""
        with self._lock:
            for cam_id in self._camera_ids:
                camera_manager.close(cam_id)
                logger.info("Closed camera %d", cam_id)
            self._camera_ids.clear()
