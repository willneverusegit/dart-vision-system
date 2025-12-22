"""
Threaded camera capture with bounded queue and graceful frame dropping.
Prevents I/O blocking and ensures real-time performance.
"""
import cv2
import threading
import queue
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from src.core import Frame

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for camera capture."""
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "any"  # "any", "dshow", "msmf" (Windows)

    def get_backend_flag(self) -> int:
        """Convert backend string to OpenCV flag."""
        backends = {
            "any": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
        }
        return backends.get(self.backend.lower(), cv2.CAP_ANY)


class ThreadedCamera:
    """
    Asynchronous camera capture using producer/consumer pattern.

    Features:
    - Runs capture in separate thread to avoid I/O blocking
    - Bounded queue with graceful frame dropping when processing is slow
    - Automatic FPS measurement
    - Thread-safe start/stop

    Example:
        camera = ThreadedCamera(queue_size=3)
        camera.start()

        while True:
            frame = camera.read()
            if frame is None:
                break
            # Process frame...

        camera.stop()
    """

    def __init__(
            self,
            config: Optional[CameraConfig] = None,
            queue_size: int = 3
    ):
        """
        Initialize threaded camera.

        Args:
            config: Camera configuration (default: CameraConfig())
            queue_size: Max frames in queue (smaller = lower latency)
        """
        self.config = config or CameraConfig()
        self.queue_size = queue_size

        # Thread-safe queue for frames
        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        # Threading control
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # OpenCV capture object
        self._capture: Optional[cv2.VideoCapture] = None

        # Statistics
        self._frame_counter = 0
        self._dropped_frames = 0
        self._last_fps_time = time.time()
        self._current_fps: float = 0.0

        # State
        self._is_running = False

    def start(self) -> bool:
        """
        Start camera capture thread.

        Returns:
            True if started successfully, False otherwise
        """
        if self._is_running:
            logger.warning("Camera already running")
            return True

        # Open camera
        backend = self.config.get_backend_flag()
        self._capture = cv2.VideoCapture(self.config.index, backend)

        if not self._capture.isOpened():
            logger.error(f"Failed to open camera {self.config.index}")
            return False

        # Set camera properties
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        # Log actual camera properties
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self._capture.get(cv2.CAP_PROP_FPS))

        logger.info(
            f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS "
            f"(requested: {self.config.width}x{self.config.height} @ {self.config.fps} FPS)"
        )

        # Start capture thread
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture"
        )
        self._capture_thread.start()

        self._is_running = True
        logger.info("Camera capture thread started")
        return True

    def stop(self) -> None:
        """Stop camera capture thread and release resources."""
        if not self._is_running:
            return

        logger.info("Stopping camera capture...")

        # Signal thread to stop
        self._stop_event.set()

        # Wait for thread to finish (with timeout)
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)

        # Release camera
        if self._capture:
            self._capture.release()
            self._capture = None

        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._is_running = False
        logger.info(
            f"Camera stopped. Stats: {self._frame_counter} frames captured, "
            f"{self._dropped_frames} frames dropped"
        )

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        """
        Read next available frame from queue.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Frame object or None if timeout/stopped
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        """Main capture loop (runs in separate thread)."""
        logger.debug("Capture loop started")

        while not self._stop_event.is_set():
            if not self._capture or not self._capture.isOpened():
                logger.error("Camera disconnected")
                break

            # Read frame
            ret, image = self._capture.read()

            if not ret or image is None:
                logger.warning("Failed to read frame")
                time.sleep(0.01)  # Brief pause before retry
                continue

            # Create Frame object
            frame = Frame(
                image=image,
                timestamp=time.time(),
                frame_id=self._frame_counter,
                fps=self._current_fps
            )

            # Update statistics
            self._frame_counter += 1
            self._update_fps()

            # Try to add to queue (non-blocking)
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                # Queue full - drop oldest frame (graceful degradation)
                try:
                    self._frame_queue.get_nowait()  # Remove oldest
                    self._frame_queue.put_nowait(frame)  # Add new
                    self._dropped_frames += 1

                    if self._dropped_frames % 10 == 0:
                        logger.warning(
                            f"Frame dropping active ({self._dropped_frames} total). "
                            "Processing too slow!"
                        )
                except queue.Empty:
                    pass

        logger.debug("Capture loop ended")

    def _update_fps(self) -> None:
        """Update FPS measurement."""
        current_time = time.time()
        elapsed = current_time - self._last_fps_time

        # Update every second
        if elapsed >= 1.0:
            self._current_fps = self._frame_counter / elapsed
            self._last_fps_time = current_time
            self._frame_counter = 0

    @property
    def is_running(self) -> bool:
        """Check if camera is currently running."""
        return self._is_running

    @property
    def fps(self) -> float:
        """Get current FPS."""
        return self._current_fps

    @property
    def dropped_frames(self) -> int:
        """Get total dropped frames."""
        return self._dropped_frames

    def get_camera_properties(self) -> dict:
        """
        Get actual camera properties.

        Returns:
            Dictionary with camera properties
        """
        if not self._capture:
            return {}

        return {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self._capture.get(cv2.CAP_PROP_FPS)),
            "backend": self._capture.getBackendName(),
        }

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()