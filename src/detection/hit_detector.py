"""
Dart hit detection with temporal confirmation.
Combines motion detection, contour analysis, and multi-frame validation.
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import time
import logging

from src.core import Frame, Hit
from src.board import DartboardMapper
from .motion import MotionDetector, MotionConfig

logger = logging.getLogger(__name__)


@dataclass
class HitDetectionConfig:
    """Configuration for hit detection."""
    # Contour filtering
    min_contour_area: int = 10  # Minimum pixels for valid dart
    max_contour_area: int = 300  # Maximum pixels
    min_aspect_ratio: float = 0.3  # Minimum width/height ratio
    max_aspect_ratio: float = 3.0  # Maximum width/height ratio

    # Temporal confirmation
    confirmation_frames: int = 5  # Hit must be stable for N frames
    position_tolerance_px: float = 10.0  # Max movement between frames
    confirmation_timeout_sec: float = 3.0  # Max time for confirmation

    # Motion detection
    motion_config: MotionConfig = field(default_factory=MotionConfig)


@dataclass
class HitCandidate:
    """A potential dart hit being tracked."""
    position: Tuple[float, float]  # (x, y) in pixels
    first_seen_frame: int
    first_seen_time: float
    confirmation_count: int = 0
    last_seen_frame: int = 0

    def is_stable(self, x: float, y: float, tolerance: float) -> bool:
        """Check if position is stable (within tolerance)."""
        dx = abs(x - self.position[0])
        dy = abs(y - self.position[1])
        return dx <= tolerance and dy <= tolerance


class HitDetector:
    """
    Detects dart hits with temporal confirmation.

    Pipeline:
    1. Motion Gating: Only process if motion detected
    2. Contour Analysis: Filter by shape (area, aspect ratio)
    3. Temporal Confirmation: Track candidates over N frames
    4. Scoring: Convert confirmed hit to score

    Example:
        detector = HitDetector(mapper)

        for frame in video_stream:
            hit = detector.detect(frame)

            if hit:
                print(f"Hit detected: {hit.score} points")
    """

    def __init__(
            self,
            mapper: DartboardMapper,
            config: Optional[HitDetectionConfig] = None
    ):
        """
        Initialize hit detector.

        Args:
            mapper: DartboardMapper for scoring
            config: Detection configuration
        """
        self.mapper = mapper
        self.config = config or HitDetectionConfig()

        # Motion detector
        self.motion_detector = MotionDetector(self.config.motion_config)

        # Candidate tracking
        self.candidates: List[HitCandidate] = []
        self.confirmed_hits: List[Hit] = []

        # Statistics
        self.frames_processed = 0
        self.motion_frames = 0
        self.candidates_created = 0
        self.hits_confirmed = 0

        logger.info(
            f"HitDetector initialized: "
            f"confirmation={self.config.confirmation_frames} frames, "
            f"tolerance={self.config.position_tolerance_px}px"
        )

    def detect(self, frame: Frame) -> Optional[Hit]:
        """
        Detect dart hit in frame.

        Args:
            frame: Input frame

        Returns:
            Hit object if confirmed, None otherwise
        """
        self.frames_processed += 1

        # Step 1: Motion Gating
        motion_mask, has_motion = self.motion_detector.detect(frame.image)

        if not has_motion:
            # No motion - clean up old candidates
            self._cleanup_stale_candidates(frame.frame_id, frame.timestamp)
            return None

        self.motion_frames += 1

        # Step 2: Contour Analysis
        contours = self._find_dart_contours(motion_mask)

        if not contours:
            return None

        # Step 3: Process candidates (temporal confirmation)
        for contour in contours:
            # ← GEÄNDERT: Use dart tip position instead of center
            cx, cy = self._get_dart_tip_position(contour)

            # Check if within board
            if not self.mapper.is_valid_hit(cx, cy):
                continue

            # Track or create candidate
            hit = self._track_candidate(
                cx, cy,
                frame.frame_id,
                frame.timestamp
            )

            if hit:
                return hit

        # Cleanup stale candidates
        self._cleanup_stale_candidates(frame.frame_id, frame.timestamp)

        return None

    def _find_dart_contours(self, motion_mask: np.ndarray) -> List[np.ndarray]:
        """
        Find contours that match dart characteristics.

        Args:
            motion_mask: Binary motion mask

        Returns:
            List of valid contours
        """
        # Find all contours
        contours, _ = cv2.findContours(
            motion_mask,
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

            valid_contours.append(contour)

        return valid_contours

    def _track_candidate(
            self,
            x: float,
            y: float,
            frame_id: int,
            timestamp: float
    ) -> Optional[Hit]:
        """
        Track candidate position and confirm if stable.

        Args:
            x: X coordinate
            y: Y coordinate
            frame_id: Current frame ID
            timestamp: Current timestamp

        Returns:
            Hit if confirmed, None otherwise
        """
        # Check if matches existing candidate
        for candidate in self.candidates:
            if candidate.is_stable(x, y, self.config.position_tolerance_px):
                # Update tracking
                candidate.confirmation_count += 1
                candidate.last_seen_frame = frame_id

                # Check if confirmed
                if candidate.confirmation_count >= self.config.confirmation_frames:
                    # Confirmed hit!
                    hit = self.mapper.pixel_to_score(
                        candidate.position[0],
                        candidate.position[1],
                        frame_id=frame_id,
                        timestamp=timestamp
                    )

                    self.hits_confirmed += 1
                    self.confirmed_hits.append(hit)

                    # Remove candidate
                    self.candidates.remove(candidate)

                    logger.info(
                        f"Hit confirmed at ({x:.1f}, {y:.1f}) after "
                        f"{candidate.confirmation_count} frames: {hit.score} points"
                    )

                    return hit

                return None

        # New candidate
        candidate = HitCandidate(
            position=(x, y),
            first_seen_frame=frame_id,
            first_seen_time=timestamp,
            confirmation_count=1,
            last_seen_frame=frame_id
        )
        self.candidates.append(candidate)
        self.candidates_created += 1

        logger.debug(f"New candidate at ({x:.1f}, {y:.1f})")

        return None

    def _cleanup_stale_candidates(self, current_frame: int, current_time: float) -> None:
        """Remove candidates that haven't been seen recently."""
        stale = []

        for candidate in self.candidates:
            # Check timeout
            time_elapsed = current_time - candidate.first_seen_time
            if time_elapsed > self.config.confirmation_timeout_sec:
                stale.append(candidate)
                continue

            # Check frame gap (in case FPS drops)
            frame_gap = current_frame - candidate.last_seen_frame
            if frame_gap > self.config.confirmation_frames * 2:
                stale.append(candidate)

        for candidate in stale:
            self.candidates.remove(candidate)
            logger.debug(f"Removed stale candidate at {candidate.position}")

    def reset(self) -> None:
        """Reset detector state."""
        self.motion_detector.reset()
        self.candidates.clear()
        logger.info("HitDetector reset")

    def get_stats(self) -> dict:
        """Get detection statistics."""
        motion_stats = self.motion_detector.get_stats()

        return {
            "frames_processed": self.frames_processed,
            "motion_frames": self.motion_frames,
            "motion_rate_percent": (
                        self.motion_frames / self.frames_processed * 100) if self.frames_processed > 0 else 0,
            "candidates_created": self.candidates_created,
            "hits_confirmed": self.hits_confirmed,
            "active_candidates": len(self.candidates),
            "confirmation_rate_percent": (
                        self.hits_confirmed / self.candidates_created * 100) if self.candidates_created > 0 else 0,
            **motion_stats
        }

    def _find_dart_contours(self, motion_mask: np.ndarray) -> List[np.ndarray]:
        """
        Find contours that match dart characteristics.

        Args:
            motion_mask: Binary motion mask

        Returns:
            List of valid contours
        """
        # Find all contours
        contours, _ = cv2.findContours(
            motion_mask,
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

            # ← NEU: Additional shape filters
            # Convexity check (dart tip should be somewhat convex)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                # Dart should have solidity > 0.3 (not too irregular)
                if solidity < 0.3:
                    continue

            # ← NEU: Perimeter check (filter out very elongated shapes)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Allow elongated shapes (dart tip) but filter extreme cases
                if circularity < 0.1:  # Very elongated
                    continue

            valid_contours.append(contour)

        return valid_contours

    def _get_dart_tip_position(self, contour: np.ndarray) -> Tuple[float, float]:
        """
        Find the dart tip position from contour.
        Uses multiple methods and returns weighted average.

        Args:
            contour: Contour to analyze

        Returns:
            (x, y) position of estimated dart tip
        """
        # Method 1: Contour center (baseline)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            # Fallback to bounding box center
            x, y, w, h = cv2.boundingRect(contour)
            return (x + w / 2, y + h / 2)

        cx_center = M["m10"] / M["m00"]
        cy_center = M["m01"] / M["m00"]

        # Method 2: Find "pointiest" part of contour
        # This is typically where the dart tip is
        hull = cv2.convexHull(contour, returnPoints=True)

        if len(hull) < 3:
            return (cx_center, cy_center)

        # Find point with highest curvature (sharpest angle)
        # This is likely the dart tip
        max_angle = 0
        tip_point = None

        for i in range(len(hull)):
            p1 = hull[i - 1][0]
            p2 = hull[i][0]
            p3 = hull[(i + 1) % len(hull)][0]

            # Calculate angle at p2
            v1 = p1 - p2
            v2 = p3 - p2

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                # Sharp angles (small angle value) indicate potential tip
                if angle < np.pi / 2 and angle > max_angle:  # Less than 90 degrees
                    max_angle = angle
                    tip_point = p2

        if tip_point is not None:
            cx_tip = float(tip_point[0])
            cy_tip = float(tip_point[1])

            # Weighted average: 30% center, 70% tip
            # This prevents outliers from tip detection
            cx = 0.3 * cx_center + 0.7 * cx_tip
            cy = 0.3 * cy_center + 0.7 * cy_tip

            return (cx, cy)

        # Fallback to center
        return (cx_center, cy_center)