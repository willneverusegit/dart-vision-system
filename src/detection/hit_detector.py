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

from src.core import Hit
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


@dataclass
class HitCandidate:
    """Candidate hit with temporal tracking."""
    position: Tuple[float, float]
    first_seen_frame: int
    first_seen_time: float
    confirmation_count: int = 0
    last_seen_frame: int = 0
    last_seen_time: float = 0.0


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

        # Frame differencing for static object detection
        self.previous_frame: Optional[np.ndarray] = None

        # Statistics
        self.frames_processed = 0
        self.motion_frames = 0
        self.candidates_created = 0
        self.hits_confirmed = 0

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

        # Step 1: Motion Detection
        motion_mask, has_motion = self.motion_detector.detect(frame.image)
        motion_pixels = cv2.countNonZero(motion_mask) if has_motion else 0

        if has_motion:
            self.motion_frames += 1

        # Step 2: Update State Machine
        confirmed_hit_pos = None
        current_state = self.state_machine.update(
            has_motion=has_motion,
            motion_pixels=motion_pixels,
            confirmed_hit=confirmed_hit_pos  # Will be set if we confirm a hit
        )

        # Step 3: Process based on state
        hit = None

        if current_state == DetectionState.CONFIRMING:
            # This is the critical phase: search for new dart
            hit = self._process_confirming_state(frame, motion_mask)

            if hit:
                # Update state machine with confirmed position
                self.state_machine.update(
                    has_motion=has_motion,
                    motion_pixels=motion_pixels,
                    confirmed_hit=(hit.x, hit.y)
                )

        # Cleanup stale candidates
        self._cleanup_stale_candidates(frame.frame_id, frame.timestamp)

        # Store frame for differencing
        self.previous_frame = frame.image.copy()

        return hit

    def _process_confirming_state(self, frame, motion_mask: np.ndarray) -> Optional[Hit]:
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
        if self.previous_frame is not None:
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

            # Track or create candidate
            hit = self._track_candidate(cx, cy, frame.frame_id, frame.timestamp)

            if hit:
                return hit

        return None

    def _find_new_objects(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray
    ) -> np.ndarray:
        """
        Find new static objects using frame differencing.

        Research: After motion stops, new dart appears as static difference.

        Args:
            current_frame: Current frame
            previous_frame: Previous frame

        Returns:
            Binary mask of new objects
        """
        # Convert to grayscale if needed
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_frame

        if len(previous_frame.shape) == 3:
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        else:
            previous_gray = previous_frame

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
        # Find matching candidate
        matched_candidate = None

        for candidate in self.candidates:
            dx = abs(x - candidate.position[0])
            dy = abs(y - candidate.position[1])
            distance = (dx**2 + dy**2)**0.5

            if distance <= self.config.position_tolerance_px:
                matched_candidate = candidate
                break

        if matched_candidate:
            # Update existing candidate
            matched_candidate.confirmation_count += 1
            matched_candidate.last_seen_frame = frame_id
            matched_candidate.last_seen_time = timestamp

            # Check if confirmed
            if matched_candidate.confirmation_count >= self.config.confirmation_frames:
                # Remove from candidates
                self.candidates.remove(matched_candidate)

                # Create hit
                hit = self.mapper.pixel_to_score(x, y)
                self.hits_confirmed += 1

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
                last_seen_time=timestamp
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
        logger.info("Detector reset")

    def get_stats(self) -> dict:
        """Get detection statistics."""
        motion_stats = self.motion_detector.get_stats()
        state_stats = self.state_machine.get_stats()

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
        }