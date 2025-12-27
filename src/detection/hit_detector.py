"""
Enhanced hit detector with state machine and sub-pixel refinement.

Research-based improvements:
- State machine for intelligent detection phases
- Hand filtering during throw phase
- Frame differencing for new object detection
- Sub-pixel refinement for precise tip location
- Temporal confirmation with position tracking
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging
from contextlib import nullcontext

from src.core import Hit
from src.core.performance_monitor import PerformanceMonitor
from src.board import DartboardMapper
from .motion import MotionDetector, MotionConfig
from .detection_state import DetectionStateMachine, StateConfig, DetectionState

logger = logging.getLogger(__name__)


@dataclass
class HitDetectionConfig:
    """Configuration for hit detection."""
    # Contour filtering
    min_contour_area: int = 10
    max_contour_area: int = 300
    min_aspect_ratio: float = 0.3
    max_aspect_ratio: float = 3.0

    # Shape filtering (research-based)
    min_solidity: float = 0.3  # Area / Convex hull area
    min_circularity: float = 0.1  # Filter very elongated shapes

    # Temporal confirmation
    confirmation_frames: int = 5
    position_tolerance_px: float = 10.0
    confirmation_timeout_sec: float = 3.0

    # Sub-pixel refinement
    enable_subpixel: bool = True
    subpixel_win_size: int = 5
    subpixel_max_iter: int = 10

    # Motion and state configs
    motion_config: MotionConfig = field(default_factory=MotionConfig)
    state_config: StateConfig = field(default_factory=StateConfig)
    background_history_frames: int = 5  # Quiet frames used for median background

    # Candidate motion analysis
    max_settling_speed_px_per_sec: float = 1.0
    velocity_settling_frames: int = 3
    reverse_motion_dot_threshold: float = -0.3

    # Spacing and plausibility
    min_frames_since_hit: int = 2
    min_pixel_drift_since_hit: float = 8.0
    entry_angle_tolerance_deg: float = 70.0

    # Performance and logging
    timing_log_interval_frames: int = 120
    enable_stage_timing: bool = True


@dataclass
class HitCandidate:
    """Candidate hit with temporal tracking."""
    position: Tuple[float, float]
    first_seen_frame: int
    first_seen_time: float
    confirmation_count: int = 0
    last_seen_frame: int = 0
    last_seen_time: float = 0.0
    positions: List[Tuple[float, float]] = field(default_factory=list)
    velocity_history: List[float] = field(default_factory=list)
    last_motion_vector: Optional[Tuple[float, float]] = None


class EnhancedHitDetector:
    """
    Enhanced hit detector with state machine and advanced features.

    Research-based pipeline:
    1. Motion Gating: Detect significant motion
    2. State Machine: Track detection phase (Idle/Watching/Confirming/Cooldown)
    3. Hand Filtering: Ignore large blobs during throw
    4. Contour Detection: Find dart-like objects with shape filters
    5. Sub-Pixel Refinement: Precise tip location
    6. Temporal Confirmation: Verify hit across multiple frames

    False positive reduction:
    - Hand filtering during WATCHING state
    - Shape-based contour filtering (solidity, circularity)
    - Temporal confirmation (5 frames)
    - Cooldown zones after confirmed hits
    """

    def __init__(
        self,
        mapper: DartboardMapper,
        config: Optional[HitDetectionConfig] = None
    ):
        """
        Initialize enhanced hit detector.

        Args:
            mapper: Dartboard mapper for scoring
            config: Detection configuration
        """
        self.mapper = mapper
        self.config = config or HitDetectionConfig()

        # Motion detector
        self.motion_detector = MotionDetector(self.config.motion_config)

        # State machine
        self.state_machine = DetectionStateMachine(self.config.state_config)

        # Candidate tracking
        self.candidates: List[HitCandidate] = []
        self.confirmed_hits: List[Hit] = []
        self.last_confirmed_hit: Optional[Hit] = None
        self.last_confirmed_frame_id: Optional[int] = None
        self.last_confirmed_time: Optional[float] = None

        # Frame differencing for static object detection
        self.previous_frame: Optional[np.ndarray] = None
        self.quiet_frame_buffer: List[np.ndarray] = []
        self.consecutive_no_motion_frames: int = 0
        # ← NEU: Store last motion mask for visualization
        self.last_motion_mask: Optional[np.ndarray] = None

        # Statistics
        self.frames_processed = 0
        self.motion_frames = 0
        self.candidates_created = 0
        self.hits_confirmed = 0

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor() if self.config.enable_stage_timing else None

        logger.info("EnhancedHitDetector initialized with state machine")

    def detect(self, frame) -> Optional[Hit]:
        """
        Detect dart hit in frame.

        Args:
            frame: Input frame with image, frame_id, timestamp

        Returns:
            Hit object if confirmed, None otherwise
        """
        self.frames_processed += 1

        timestamp = getattr(frame, "timestamp", None)
        frame_id = getattr(frame, "frame_id", 0) or 0

        # Step 0: Preprocess (placeholder for future filters)
        with self._measure_stage("preprocess"):
            processed_image = frame.image

        # Step 1: Motion Detection
        with self._measure_stage("motion"):
            motion_mask, has_motion = self.motion_detector.detect(
                processed_image,
                timestamp=timestamp,
                frame_id=frame_id
            )
        # ← NEU: Store for visualization
        self.last_motion_mask = motion_mask
        motion_pixels = cv2.countNonZero(motion_mask) if has_motion else 0

        if has_motion:
            self.motion_frames += 1
            self.consecutive_no_motion_frames = 0
        else:
            self.consecutive_no_motion_frames += 1
            self._update_quiet_background(frame.image)

        # Step 2: Update State Machine
        confirmed_hit_pos = None
        current_state = self.state_machine.update(
            has_motion=has_motion,
            motion_pixels=motion_pixels,
            confirmed_hit=confirmed_hit_pos,  # Will be set if we confirm a hit
            frame_id=frame.frame_id
        )

        # Step 3: Process based on state
        hit = None

        if current_state == DetectionState.CONFIRMING:
            quiet_needed = getattr(self.config.state_config, "confirming_quiet_frames", 0)
            if quiet_needed and self.consecutive_no_motion_frames < quiet_needed:
                logger.debug(
                    "Waiting for quiet frames before confirming: %d/%d",
                    self.consecutive_no_motion_frames,
                    quiet_needed
                )
            else:
                # This is the critical phase: search for new dart
                background_frame = self._get_background_frame()
                with self._measure_stage("contour"):
                    hit = self._process_confirming_state(
                        frame,
                        motion_mask,
                        background_frame=background_frame
                    )

            if hit:
                # Update state machine with confirmed position
                self.state_machine.update(
                    has_motion=has_motion,
                    motion_pixels=motion_pixels,
                    confirmed_hit=(hit.x, hit.y),
                    frame_id=frame.frame_id
                )

        # Cleanup stale candidates
        self._cleanup_stale_candidates(frame_id, timestamp or time.time())

        # Store frame for differencing
        self.previous_frame = processed_image.copy()

        # Periodic timing log for visibility
        self._maybe_log_timing(frame_id)

        return hit

    def _process_confirming_state(
        self,
        frame,
        motion_mask: np.ndarray,
        background_frame: Optional[np.ndarray] = None
    ) -> Optional[Hit]:
        """
        Process frame in CONFIRMING state.

        Uses frame differencing to find new static objects.

        Args:
            frame: Input frame
            motion_mask: Current motion mask

        Returns:
            Confirmed hit or None
        """
        # Frame differencing: Find new objects
        if background_frame is not None:
            new_objects_mask = self._find_new_objects(frame.image, background_frame)
        elif self.previous_frame is not None:
            new_objects_mask = self._find_new_objects(frame.image, self.previous_frame)
        else:
            new_objects_mask = motion_mask

        # Find contours in new objects
        contours = self._find_dart_contours(new_objects_mask)

        if not contours:
            return None

        # Process each contour
        for contour in contours:
            # Get refined tip position
            cx, cy = self._get_dart_tip_position(contour, frame.image)

            # Check if within board
            if not self.mapper.is_valid_hit(cx, cy):
                continue

            # Check if in cooldown zone
            if self.state_machine.is_in_cooldown_zone(cx, cy):
                logger.debug(f"Position ({cx:.0f}, {cy:.0f}) in cooldown zone")
                continue

            # Additional gating vs last confirmed hit
            if not self._passes_last_hit_constraints(cx, cy, frame.frame_id):
                continue

            # Track or create candidate
            hit = self._track_candidate(cx, cy, frame.frame_id, frame.timestamp)

            if hit:
                return hit

        return None

    def _find_matching_candidate(self, x: float, y: float) -> Optional[HitCandidate]:
        """Find existing candidate within tolerance."""
        for candidate in self.candidates:
            dx = abs(x - candidate.position[0])
            dy = abs(y - candidate.position[1])
            distance = (dx ** 2 + dy ** 2) ** 0.5

            if distance <= self.config.position_tolerance_px:
                return candidate

        return None

    def _is_reverse_motion(
        self,
        previous_vector: Optional[Tuple[float, float]],
        current_vector: Tuple[float, float]
    ) -> bool:
        """Check if motion reverses direction (likely noise/reflection)."""
        if previous_vector is None:
            return False

        prev_norm = (previous_vector[0] ** 2 + previous_vector[1] ** 2) ** 0.5
        curr_norm = (current_vector[0] ** 2 + current_vector[1] ** 2) ** 0.5
        if prev_norm == 0 or curr_norm == 0:
            return False

        cos_sim = (
            (previous_vector[0] * current_vector[0]) +
            (previous_vector[1] * current_vector[1])
        ) / (prev_norm * curr_norm)

        return cos_sim < self.config.reverse_motion_dot_threshold

    def _has_settled_speed(self, candidate: HitCandidate) -> bool:
        """
        Require velocity to trend toward zero before confirming.

        Prevents sliding/blurring artifacts from being accepted.
        """
        if not candidate.velocity_history:
            return True

        sample_len = min(
            self.config.velocity_settling_frames,
            len(candidate.velocity_history)
        )

        recent = candidate.velocity_history[-sample_len:]
        return max(recent) <= self.config.max_settling_speed_px_per_sec

    def _passes_entry_angle(self, candidate: HitCandidate) -> bool:
        """
        Ensure candidate path points toward board center (plausible entry).

        Uses simple line fit between first and latest positions.
        """
        if len(candidate.positions) < 2:
            return True

        start = candidate.positions[0]
        end = candidate.positions[-1]
        approach_vector = (end[0] - start[0], end[1] - start[1])

        center_vector = (
            self.mapper.center[0] - end[0],
            self.mapper.center[1] - end[1]
        )

        approach_norm = (approach_vector[0] ** 2 + approach_vector[1] ** 2) ** 0.5
        center_norm = (center_vector[0] ** 2 + center_vector[1] ** 2) ** 0.5

        if approach_norm == 0 or center_norm == 0:
            return True

        cos_sim = (
            (approach_vector[0] * center_vector[0]) +
            (approach_vector[1] * center_vector[1])
        ) / (approach_norm * center_norm)

        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_sim))
        return angle <= self.config.entry_angle_tolerance_deg

    def _can_accept_hit(self, hit: Hit, frame_id: int) -> bool:
        """
        Enforce spacing and plausibility before emitting hit.
        """
        if self.last_confirmed_hit:
            if (
                self.last_confirmed_frame_id is not None and
                (frame_id - self.last_confirmed_frame_id) < self.config.min_frames_since_hit
            ):
                logger.debug("Hit rejected: too few frames since last hit")
                return False

            drift = (
                (hit.x_px - self.last_confirmed_hit.x_px) ** 2 +
                (hit.y_px - self.last_confirmed_hit.y_px) ** 2
            ) ** 0.5
            if drift < self.config.min_pixel_drift_since_hit:
                logger.debug("Hit rejected: insufficient pixel drift since last hit")
                return False

        return True

    def _passes_last_hit_constraints(
            self,
            x: float,
            y: float,
            frame_id: int
    ) -> bool:
        """
        Gate new candidates against last confirmed hit position/ring.
        """
        if self.last_confirmed_hit is None:
            return True

        # Temporal spacing
        if self.last_confirmed_frame_id is not None:
            if (frame_id - self.last_confirmed_frame_id) < self.config.min_frames_since_hit:
                logger.debug("Candidate discarded: within min frame spacing of last hit")
                return False

        # Spatial/ring spacing
        distance = (
            (x - self.last_confirmed_hit.x_px) ** 2 +
            (y - self.last_confirmed_hit.y_px) ** 2
        ) ** 0.5
        if distance < max(
            self.state_machine.config.cooldown_radius_px,
            self.config.min_pixel_drift_since_hit
        ):
            logger.debug("Candidate discarded: within cooldown/spacing radius")
            return False

        candidate_ring, candidate_radius = self.mapper.ring_for_position(x, y)
        last_ring, last_radius = self.mapper.ring_for_position(
            self.last_confirmed_hit.x_px,
            self.last_confirmed_hit.y_px
        )
        if (
            candidate_ring == last_ring and
            abs(candidate_radius - last_radius) < self.config.min_pixel_drift_since_hit
        ):
            logger.debug("Candidate discarded: same ring without sufficient radial change")
            return False

        # Entry angle plausibility relative to radial direction
        radius, angle = self.mapper.pixel_to_polar(x, y)
        _, last_angle = self.mapper.pixel_to_polar(
            self.last_confirmed_hit.x_px,
            self.last_confirmed_hit.y_px
        )
        angle_diff = abs((angle - last_angle + 180) % 360 - 180)
        if angle_diff > self.config.entry_angle_tolerance_deg * 2:
            logger.debug("Candidate discarded: angle too far from prior entry corridor")
            return False

        return True
    def _find_new_objects(
        self,
        current_frame: np.ndarray,
        background_frame: np.ndarray
    ) -> np.ndarray:
        """
        Find new static objects using frame differencing.

        Research: After motion stops, new dart appears as static difference.

        Args:
            current_frame: Current frame
            background_frame: Background frame (median of quiet frames or last frame)

        Returns:
            Binary mask of new objects
        """
        current_gray = self._to_grayscale(current_frame)
        previous_gray = self._to_grayscale(background_frame)

        # Absolute difference
        diff = cv2.absdiff(current_gray, previous_gray)

        # Threshold
        _, diff_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        # Morphology to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)

        return diff_mask

    def _find_dart_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Find contours matching dart characteristics.

        Research-based filtering:
        - Size: 10-300 px²
        - Aspect ratio: 0.3-3.0
        - Solidity: >0.3 (not too irregular)
        - Circularity: >0.1 (filter extreme elongation)

        Args:
            mask: Binary mask

        Returns:
            List of valid contours
        """
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        valid_contours = []

        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area or area > self.config.max_contour_area:
                continue

            # Filter by aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue

            aspect_ratio = w / h
            if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
                continue

            # Solidity check (research-based)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < self.config.min_solidity:
                    continue

            # Circularity check (research-based)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < self.config.min_circularity:
                    continue

            valid_contours.append(contour)

        return valid_contours

    def _get_dart_tip_position(
        self,
        contour: np.ndarray,
        image: np.ndarray
    ) -> Tuple[float, float]:
        """
        Find dart tip position with sub-pixel refinement.

        Research approach:
        1. Find contour moments (baseline center)
        2. Find sharpest point on convex hull (likely tip)
        3. Optional: Sub-pixel corner refinement
        4. Weighted average for robustness

        Args:
            contour: Contour to analyze
            image: Source image for sub-pixel refinement

        Returns:
            (x, y) position of dart tip
        """
        # Method 1: Contour center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            x, y, w, h = cv2.boundingRect(contour)
            return (x + w/2, y + h/2)

        cx_center = M["m10"] / M["m00"]
        cy_center = M["m01"] / M["m00"]

        # Method 2: Find sharpest point (tip)
        hull = cv2.convexHull(contour, returnPoints=True)

        if len(hull) < 3:
            return (cx_center, cy_center)

        # Find point with sharpest angle
        min_angle = np.pi
        tip_point = None

        for i in range(len(hull)):
            p1 = hull[i-1][0]
            p2 = hull[i][0]
            p3 = hull[(i+1) % len(hull)][0]

            v1 = p1 - p2
            v2 = p3 - p2

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                if angle < min_angle:
                    min_angle = angle
                    tip_point = p2

        if tip_point is None:
            return (cx_center, cy_center)

        cx_tip = float(tip_point[0])
        cy_tip = float(tip_point[1])

        # Method 3: Sub-pixel refinement (research-based)
        if self.config.enable_subpixel:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Prepare corner point
            corners = np.array([[cx_tip, cy_tip]], dtype=np.float32)

            # Sub-pixel refinement
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                self.config.subpixel_max_iter,
                0.001
            )

            try:
                refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (self.config.subpixel_win_size, self.config.subpixel_win_size),
                    (-1, -1),
                    criteria
                )

                if refined is not None:
                    cx_tip = float(refined[0][0])
                    cy_tip = float(refined[0][1])
            except cv2.error:
                # Sub-pixel failed, use original
                pass

        # Weighted average: 20% center, 80% tip
        # (Research: tip is more accurate but can be noisy)
        cx = 0.2 * cx_center + 0.8 * cx_tip
        cy = 0.2 * cy_center + 0.8 * cy_tip

        return (cx, cy)

    def _track_candidate(
        self,
        x: float,
        y: float,
        frame_id: int,
        timestamp: float
    ) -> Optional[Hit]:
        """
        Track hit candidate with temporal confirmation.

        Args:
            x, y: Hit position
            frame_id: Current frame ID
            timestamp: Current timestamp

        Returns:
            Hit if confirmed, None otherwise
        """
        matched_candidate = self._find_matching_candidate(x, y)

        if matched_candidate:
            # Motion analysis
            motion_vector = (
                x - matched_candidate.position[0],
                y - matched_candidate.position[1]
            )
            dt = max(timestamp - matched_candidate.last_seen_time, 1e-3)
            speed = (motion_vector[0] ** 2 + motion_vector[1] ** 2) ** 0.5 / dt

            if self._is_reverse_motion(matched_candidate.last_motion_vector, motion_vector):
                logger.debug("Candidate rejected due to reverse motion")
                self.candidates.remove(matched_candidate)
                return None

            matched_candidate.velocity_history.append(speed)
            matched_candidate.positions.append((x, y))
            matched_candidate.last_motion_vector = motion_vector

            # Update positional info
            matched_candidate.position = (x, y)
            matched_candidate.confirmation_count += 1
            matched_candidate.last_seen_frame = frame_id
            matched_candidate.last_seen_time = timestamp

            if not self._has_settled_speed(matched_candidate):
                return None

            if not self._passes_entry_angle(matched_candidate):
                return None

            if matched_candidate.confirmation_count >= self.config.confirmation_frames:
                # Create hit
                with self._measure_stage("scoring"):
                    hit = self.mapper.pixel_to_score(x, y, frame_id=frame_id, timestamp=timestamp)

                if not self._can_accept_hit(hit, frame_id):
                    # Remove noisy candidate
                    self.candidates.remove(matched_candidate)
                    return None

                self.candidates.remove(matched_candidate)
                self.hits_confirmed += 1
                self.confirmed_hits.append(hit)
                self.last_confirmed_hit = hit
                self.last_confirmed_frame_id = frame_id
                self.last_confirmed_time = timestamp

                logger.info(
                    f"Hit confirmed: {hit.score} points at ({x:.1f}, {y:.1f}) "
                    f"after {matched_candidate.confirmation_count} frames"
                )

                return hit

        else:
            # Create new candidate
            candidate = HitCandidate(
                position=(x, y),
                first_seen_frame=frame_id,
                first_seen_time=timestamp,
                confirmation_count=1,
                last_seen_frame=frame_id,
                last_seen_time=timestamp,
                positions=[(x, y)],
            )
            self.candidates.append(candidate)
            self.candidates_created += 1

            logger.debug(f"New candidate at ({x:.1f}, {y:.1f})")

        return None

    def _cleanup_stale_candidates(self, current_frame: int, current_time: float) -> None:
        """Remove stale candidates that timed out."""
        timeout = self.config.confirmation_timeout_sec

        self.candidates = [
            c for c in self.candidates
            if (current_time - c.last_seen_time) < timeout
        ]

    def reset(self) -> None:
        """Reset detector state."""
        self.motion_detector.reset()
        self.state_machine.reset()
        self.candidates.clear()
        self.previous_frame = None
        self.quiet_frame_buffer.clear()
        self.consecutive_no_motion_frames = 0
        self.last_confirmed_hit = None
        self.last_confirmed_frame_id = None
        self.last_confirmed_time = None
        self.confirmed_hits.clear()
        if self.performance_monitor:
            self.performance_monitor.reset()
        logger.info("Detector reset")

    def get_stats(self) -> dict:
        """Get detection statistics."""
        motion_stats = self.motion_detector.get_stats()
        state_stats = self.state_machine.get_stats()
        timing_report = self.get_performance_report()

        confirmation_rate = 0.0
        if self.candidates_created > 0:
            confirmation_rate = (self.hits_confirmed / self.candidates_created) * 100.0

        return {
            **motion_stats,
            **state_stats,
            "candidates_created": self.candidates_created,
            "hits_confirmed": self.hits_confirmed,
            "confirmation_rate_percent": confirmation_rate,
            "active_candidates": len(self.candidates),
            "last_confirmed_frame_id": self.last_confirmed_frame_id,
            "last_confirmed_time": self.last_confirmed_time,
            "quiet_frame_buffer": len(self.quiet_frame_buffer),
            "consecutive_no_motion_frames": self.consecutive_no_motion_frames,
            "timing_ms": timing_report,
        }

    def _update_quiet_background(self, frame: np.ndarray) -> None:
        """Store quiet frames for background estimation."""
        if self.config.background_history_frames <= 0:
            return

        gray_frame = self._to_grayscale(frame).copy()
        self.quiet_frame_buffer.append(gray_frame)

        if len(self.quiet_frame_buffer) > self.config.background_history_frames:
            self.quiet_frame_buffer.pop(0)

    def _get_background_frame(self) -> Optional[np.ndarray]:
        """Compute median background from quiet frames."""
        if not self.quiet_frame_buffer:
            return None

        stack = np.stack(self.quiet_frame_buffer, axis=0)
        median_background = np.median(stack, axis=0).astype(np.uint8)
        return median_background

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale if needed."""
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def get_performance_report(self) -> dict:
        """Expose performance timings for pipeline stages."""
        if not self.performance_monitor:
            return {}
        return self.performance_monitor.get_report()

    def _measure_stage(self, name: str):
        """Context manager for stage timing with graceful disable."""
        if self.performance_monitor:
            return self.performance_monitor.measure(name)
        return nullcontext()

    def _maybe_log_timing(self, frame_id: int) -> None:
        """Periodically log stage timings and adaptive thresholds."""
        interval = getattr(self.config, "timing_log_interval_frames", 0)
        if not interval or not self.performance_monitor:
            return

        if frame_id <= 0 or frame_id % interval != 0:
            return

        report = self.performance_monitor.get_report()
        if not report:
            return

        stage_summary = ", ".join(
            f"{name}:{stats.get('recent_avg_ms', 0.0):.2f}ms"
            for name, stats in sorted(report.items())
        )
        motion_stats = self.motion_detector.get_stats()
        logger.info(
            "Pipeline timing @frame %d → %s | fps=%.1f motion=%.1f%% varThr=%.1f minArea=%d",
            frame_id,
            stage_summary,
            motion_stats.get("fps_estimate", 0.0),
            motion_stats.get("motion_rate_percent", 0.0),
            motion_stats.get("var_threshold", 0.0),
            motion_stats.get("min_motion_area", 0),
        )
