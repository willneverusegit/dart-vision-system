"""
Threaded camera capture with bounded queue and graceful frame dropping.
Supports both live camera and video files.
"""
import cv2
import threading
import queue
import time
import logging
from typing import Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

from src.core import Frame

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for camera/video capture."""
    source: Union[int, str, Path] = 0  # Camera index or video file path
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "any"  # "any", "dshow", "msmf" (Windows)
    loop_video: bool = True  # Loop video files when reaching end

    def __post_init__(self):
        """Convert Path to string."""
        if isinstance(self.source, Path):
            self.source = str(self.source)

    def is_video_file(self) -> bool:
        """Check if source is a video file (vs camera)."""
        return isinstance(self.source, str) and Path(self.source).exists()

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
    Asynchronous camera/video capture using producer/consumer pattern.

    Supports:
    - Live camera capture (source = camera index)
    - Video file playback (source = file path)

    Features:
    - Runs capture in separate thread to avoid I/O blocking
    - Bounded queue with graceful frame dropping when processing is slow
    - Automatic FPS measurement
    - Video looping for continuous testing
    - Thread-safe start/stop

    Example (Camera):
        camera = ThreadedCamera(CameraConfig(source=0))
        camera.start()

    Example (Video):
        camera = ThreadedCamera(CameraConfig(source="videos/test.mp4"))
        camera.start()
    """

    def __init__(
            self,
            config: Optional[CameraConfig] = None,
            queue_size: int = 3
    ):
        """
        Initialize threaded camera/video capture.

        Args:
            config: Camera/video configuration (default: CameraConfig())
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

        # Video-specific
        self._is_video_file = self.config.is_video_file()
        self._video_frame_count = 0
        self._video_total_frames = 0

        # State
        self._is_running = False

        # ← NEU: Für Log-Throttling
        self._consecutive_read_failures = 0
        self._last_failure_log_time = 0.0

    def start(self) -> bool:
        """
        Start camera/video capture thread.

        Returns:
            True if started successfully, False otherwise
        """
        if self._is_running:
            logger.warning("Capture already running")
            return True

        # Open camera or video file
        if self._is_video_file:
            self._capture = cv2.VideoCapture(str(self.config.source))
            source_name = Path(self.config.source).name
        else:
            backend = self.config.get_backend_flag()
            self._capture = cv2.VideoCapture(self.config.source, backend)
            source_name = f"Camera {self.config.source}"

        if not self._capture.isOpened():
            logger.error(f"Failed to open {source_name}")
            return False

        # Set camera properties (only for cameras, not video files)
        if not self._is_video_file:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        # Get actual properties
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self._capture.get(cv2.CAP_PROP_FPS))

        # Video-specific info
        if self._is_video_file:
            self._video_total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(
                f"Video opened: {source_name} - "
                f"{actual_width}x{actual_height} @ {actual_fps} FPS - "
                f"{self._video_total_frames} frames total"
            )
        else:
            logger.info(
                f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS "
                f"(requested: {self.config.width}x{self.config.height} @ {self.config.fps} FPS)"
            )

        # Start capture thread
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture" if not self._is_video_file else "VideoCapture"
        )
        self._capture_thread.start()

        self._is_running = True
        logger.info(f"{'Video' if self._is_video_file else 'Camera'} capture thread started")
        return True

    def stop(self) -> None:
        """Stop capture thread and release resources."""
        if not self._is_running:
            return

        logger.info("Stopping capture...")

        # Signal thread to stop
        self._stop_event.set()

        # Wait for thread to finish (with timeout)
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)

        # Release camera/video
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
            f"Capture stopped. Stats: {self._frame_counter} frames captured, "
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
                logger.error("Capture source disconnected")
                break

            # Read frame
            ret, image = self._capture.read()

            # Handle video end
            if not ret or image is None:
                self._consecutive_read_failures += 1

                if self._is_video_file and self.config.loop_video:
                    # Loop video
                    current_time = time.time()
                    # Log only first failure or once per second
                    if (self._consecutive_read_failures == 1 or
                            current_time - self._last_failure_log_time > 1.0):
                        logger.debug("Video ended, looping...")
                        self._last_failure_log_time = current_time

                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._video_frame_count = 0
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                else:
                    # Log warning only once or every 100 failures
                    if self._consecutive_read_failures == 1 or self._consecutive_read_failures % 100 == 0:
                        logger.warning(
                            f"Failed to read frame or video ended "
                            f"({self._consecutive_read_failures} consecutive failures)"
                        )

                    if not self._is_video_file:
                        time.sleep(0.01)  # Brief pause before retry
                    else:
                        # Video ended and not looping - stop trying
                        time.sleep(0.1)  # Longer pause to avoid CPU spin
                    continue

            # Successful read - reset failure counter
            self._consecutive_read_failures = 0

            # Track video progress
            if self._is_video_file:
                self._video_frame_count += 1

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
        """Check if capture is currently running."""
        return self._is_running

    @property
    def is_video(self) -> bool:
        """Check if source is a video file."""
        return self._is_video_file

    @property
    def fps(self) -> float:
        """Get current FPS."""
        return self._current_fps

    @property
    def dropped_frames(self) -> int:
        """Get total dropped frames."""
        return self._dropped_frames

    @property
    def video_progress(self) -> Optional[Tuple[int, int]]:
        """
        Get video playback progress.

        Returns:
            (current_frame, total_frames) or None if not a video
        """
        if not self._is_video_file:
            return None
        return (self._video_frame_count, self._video_total_frames)

    def get_capture_properties(self) -> dict:
        """
        Get actual capture properties.

        Returns:
            Dictionary with capture properties
        """
        if not self._capture:
            return {}

        props = {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self._capture.get(cv2.CAP_PROP_FPS)),
            "backend": self._capture.getBackendName(),
            "is_video": self._is_video_file,
        }

        if self._is_video_file:
            props["total_frames"] = self._video_total_frames
            props["current_frame"] = self._video_frame_count

        return props

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()