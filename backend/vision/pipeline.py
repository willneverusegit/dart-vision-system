"""Contour pipeline, dart tip extraction, and detection loop."""

import base64
import logging
import threading
import time
from collections.abc import Callable
from enum import StrEnum

import cv2
import numpy as np
from pydantic import BaseModel, Field

from backend.models.board import BoardModel
from backend.models.game import HitEvent
from backend.scoring.board_model import board_to_polar, get_score, pixel_to_board, polar_to_field
from backend.vision.camera import camera_manager
from backend.vision.detection import BackgroundModel, detect_motion, wait_for_stable_frame

logger = logging.getLogger(__name__)


class PipelineMode(StrEnum):
    """Pipeline processing mode."""

    NORMAL = "normal"
    ECO = "eco"
    DEBUG = "debug"


class DebugOutput(BaseModel):
    """Debug thumbnails from each pipeline stage."""

    grayscale_b64: str | None = None
    diff_b64: str | None = None
    canny_b64: str | None = None
    contours_b64: str | None = None


class PipelineResult(BaseModel):
    """Result of processing a single frame through the detection pipeline."""

    tip_point: tuple[int, int] | None = None
    field: str = ""
    score: int = 0
    multiplier: int = 1
    contour_count: int = 0
    debug_frame_b64: str | None = Field(default=None, exclude=True)
    debug_output: DebugOutput | None = None


def process_frame(
    frame: np.ndarray,
    background: np.ndarray,
    board: BoardModel | None = None,
    mode: PipelineMode = PipelineMode.NORMAL,
) -> PipelineResult:
    """Full detection pipeline: diff -> Canny -> contour filter -> tip extraction.

    Args:
        frame: Current BGR frame with dart.
        background: Grayscale background (empty board).
        board: BoardModel for coordinate mapping (optional).
        mode: Pipeline mode (normal, eco, debug).

    Returns:
        PipelineResult with detected tip and score.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    diff = cv2.absdiff(gray, background)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    if mode == PipelineMode.ECO:
        # Eco mode: skip Canny, use thresholded diff directly for contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edges = None
    else:
        # Normal/Debug: full Canny pipeline
        edges = cv2.Canny(thresh, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours: dart-shaped (elongated, reasonable area)
    dart_contours = _filter_dart_contours(contours)

    tip = extract_tip(dart_contours)

    result = PipelineResult(
        tip_point=tip,
        contour_count=len(dart_contours),
    )

    if tip is not None and board is not None:
        homography = np.array(board.homography, dtype=np.float64)
        board_pt = pixel_to_board(tip, homography)
        radius, angle = board_to_polar(board_pt)
        field_name = polar_to_field(radius, angle, board)
        base_score, multiplier = get_score(field_name)

        result.field = field_name
        result.score = base_score * multiplier
        result.multiplier = multiplier

    # Debug output: generate thumbnails of each stage
    if mode == PipelineMode.DEBUG:
        result.debug_output = _build_debug_output(gray, diff, edges, dart_contours, frame.shape[:2])

    return result


def _encode_thumbnail(img: np.ndarray, size: int = 160) -> str:
    """Encode an image as base64 JPEG thumbnail."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
    if len(thumb.shape) == 2:
        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _build_debug_output(
    gray: np.ndarray,
    diff: np.ndarray,
    edges: np.ndarray | None,
    contours: list[np.ndarray],
    frame_shape: tuple[int, int],
) -> DebugOutput:
    """Build debug thumbnails from pipeline stages."""
    # Contour visualization
    contour_img = np.zeros((*frame_shape, 3), dtype=np.uint8)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    return DebugOutput(
        grayscale_b64=_encode_thumbnail(gray),
        diff_b64=_encode_thumbnail(diff),
        canny_b64=_encode_thumbnail(edges) if edges is not None else None,
        contours_b64=_encode_thumbnail(contour_img),
    )


def _filter_dart_contours(
    contours: list[np.ndarray],
    min_area: int = 200,
    max_area: int = 50000,
    min_aspect: float = 1.5,
) -> list[np.ndarray]:
    """Filter contours to keep only dart-shaped ones."""
    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect >= min_aspect:
            filtered.append(c)
    return filtered


def extract_tip(contours: list[np.ndarray]) -> tuple[int, int] | None:
    """Find the dart tip from filtered contours.

    Uses the lowest point (highest y-value) of the largest contour,
    assuming the dart enters from top/side and tip points toward the board.
    If multiple contours, merges them and finds the extreme point closest
    to the centroid direction.
    """
    if not contours:
        return None

    # Merge all dart contours
    all_points = np.vstack(contours)

    # Find the topmost point (lowest y = tip for a dart coming from above)
    # and bottommost (highest y = tip for dart from below)
    # Use the point furthest from centroid as tip
    M = cv2.moments(all_points)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Find the point most distant from centroid
    max_dist = 0
    tip = None
    for pt in all_points:
        px, py = pt[0]
        dist = (px - cx) ** 2 + (py - cy) ** 2
        if dist > max_dist:
            max_dist = dist
            tip = (int(px), int(py))

    return tip


def create_debug_frame(
    frame: np.ndarray,
    background: np.ndarray,
    result: PipelineResult,
) -> bytes:
    """Create an annotated debug frame as JPEG bytes."""
    debug = frame.copy()

    if result.tip_point is not None:
        tx, ty = result.tip_point
        cv2.circle(debug, (tx, ty), 8, (0, 0, 255), 2)
        cv2.circle(debug, (tx, ty), 2, (0, 0, 255), -1)
        label = f"{result.field} ({result.score})"
        cv2.putText(debug, label, (tx + 12, ty - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    _, buf = cv2.imencode(".jpg", debug, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


class DetectionLoop:
    """Background thread that continuously detects dart hits."""

    def __init__(
        self,
        camera_id: int,
        background_model: BackgroundModel,
        board_model: BoardModel | None = None,
        on_hit: Callable[[HitEvent, bytes | None], None] | None = None,
        debug: bool = False,
        poll_interval: float = 0.05,
        mode: PipelineMode = PipelineMode.NORMAL,
    ) -> None:
        self._camera_id = camera_id
        self._background_model = background_model
        self._board_model = board_model
        self._on_hit = on_hit
        self._debug = debug
        self._poll_interval = poll_interval
        self._mode = mode
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_counter = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the detection loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Detection loop started for camera %d", self._camera_id)

    def stop(self) -> None:
        """Stop the detection loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Detection loop stopped for camera %d", self._camera_id)

    def _loop(self) -> None:
        """Main detection loop running in background thread."""
        bg = self._background_model.get_background()
        if bg is None:
            logger.error("No background set, cannot start detection")
            self._running = False
            return

        while self._running:
            frame = camera_manager.capture_frame(self._camera_id)
            if frame is None:
                time.sleep(self._poll_interval)
                continue

            # Eco mode: skip 2 of every 3 frames
            self._frame_counter += 1
            if self._mode == PipelineMode.ECO and self._frame_counter % 3 != 0:
                time.sleep(self._poll_interval)
                continue

            if not detect_motion(frame, bg):
                time.sleep(self._poll_interval)
                continue

            # Motion detected — wait for dart to settle
            stable = wait_for_stable_frame(self._camera_id, delay_ms=300)
            if stable is None:
                continue

            effective_mode = PipelineMode.DEBUG if self._debug else self._mode
            result = process_frame(stable, bg, self._board_model, mode=effective_mode)
            if result.tip_point is None:
                continue

            # Create HitEvent
            hit = HitEvent(
                camera_ids=[str(self._camera_id)],
                image_points={str(self._camera_id): result.tip_point},
                board_point=result.tip_point,
                field=result.field,
                score=result.score,
                multiplier=result.multiplier,
            )

            debug_jpeg = None
            if self._debug:
                debug_jpeg = create_debug_frame(stable, bg, result)

            if self._on_hit:
                self._on_hit(hit, debug_jpeg)

            # Wait a bit before next detection to avoid double-counting
            time.sleep(1.0)
