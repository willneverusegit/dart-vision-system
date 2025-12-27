"""
Simple, robust hit detector - focus on what works.

Philosophy:
1. Detect motion (throw happening)
2. When motion stops → find contours
3. Find dart tip
4. Done!

No over-engineering, no excessive validation.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
import time
import logging

from src.core import Hit
from src.board import DartboardMapper
from .motion import MotionDetector, MotionConfig

logger = logging.getLogger(__name__)


class SimpleHitDetector:
    """
    Simplified hit detector - back to basics.

    Research shows: Simple approaches often work better than over-engineered ones.

    Flow:
    1. Motion detected? → Start watching
    2. Motion stopped? → Look for new contours
    3. Find dart-like contour → Get tip position → Score it
    4. Cooldown briefly to avoid duplicates

    That's it!
    """

    def __init__(
        self,
        mapper: DartboardMapper,
        motion_config: Optional[MotionConfig] = None
    ):
        """
        Initialize simple detector.

        Args:
            mapper: Dartboard mapper for scoring
            motion_config: Optional motion detection config
        """
        self.mapper = mapper
        self.motion_detector = MotionDetector(motion_config or MotionConfig())

        # Simple state: are we looking for a dart?
        self.looking_for_dart = False
        self.motion_stopped_time: Optional[float] = None
        self.last_motion_time: Optional[float] = None

        # Last hit tracking (simple cooldown)
        self.last_hit: Optional[Hit] = None
        self.last_hit_time: Optional[float] = None

        # Background for difference detection
        self.background_frame: Optional[np.ndarray] = None
        self.quiet_frames: List[np.ndarray] = []

        # Config (simple!)
        self.min_contour_area = 10
        self.max_contour_area = 500
        self.cooldown_seconds = 1.0  # Ignore hits for 1 sec after confirmed hit
        self.search_duration = 1.5   # Look for dart for 1.5 seconds after motion stops
        self.quiet_frames_needed = 3  # Build background from 3 quiet frames

        logger.info("SimpleHitDetector initialized - back to basics!")

    def detect(self, frame) -> Optional[Hit]:
        """
        Detect dart hit - simple and effective.

        Args:
            frame: Input frame

        Returns:
            Hit if found, None otherwise
        """
        current_time = time.time()

        # Convert to grayscale
        gray = self._to_grayscale(frame.image)

        # Step 1: Motion detection
        motion_mask, has_motion = self.motion_detector.detect(
            frame.image,
            timestamp=current_time,
            frame_id=getattr(frame, 'frame_id', 0),
            gray_image=gray
        )

        # Step 2: State tracking (simple!)
        if has_motion:
            self.last_motion_time = current_time
            if not self.looking_for_dart:
                logger.debug("Motion detected - dart incoming!")
                self.looking_for_dart = True
                self.motion_stopped_time = None
        else:
            # No motion - collect quiet frames for background
            self.quiet_frames.append(gray.copy())
            if len(self.quiet_frames) > self.quiet_frames_needed:
                self.quiet_frames.pop(0)

            # If we were seeing motion, it just stopped
            if self.looking_for_dart and self.motion_stopped_time is None:
                self.motion_stopped_time = current_time
                logger.debug("Motion stopped - searching for dart...")

        # Step 3: Are we in cooldown?
        if self.last_hit_time and (current_time - self.last_hit_time) < self.cooldown_seconds:
            return None  # Too soon after last hit

        # Step 4: Should we search for dart?
        if not self.looking_for_dart or self.motion_stopped_time is None:
            return None  # Not the right time

        time_since_motion_stopped = current_time - self.motion_stopped_time
        if time_since_motion_stopped > self.search_duration:
            # Timeout - stop looking
            logger.debug("Search timeout - no dart found")
            self.looking_for_dart = False
            self.motion_stopped_time = None
            return None

        # Step 5: Build background if we have enough quiet frames
        if len(self.quiet_frames) >= self.quiet_frames_needed:
            self.background_frame = np.median(
                np.stack(self.quiet_frames), axis=0
            ).astype(np.uint8)

        # Step 6: Find new objects (dart!)
        if self.background_frame is None:
            return None  # Need background first

        # Difference from background
        diff = cv2.absdiff(gray, self.background_frame)
        _, diff_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)

        # Step 7: Find contours
        contours, _ = cv2.findContours(
            diff_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Step 8: Filter contours by size (keep it simple!)
        valid_contours = [
            c for c in contours
            if self.min_contour_area <= cv2.contourArea(c) <= self.max_contour_area
        ]

        if not valid_contours:
            return None

        # Step 9: Take the best contour (largest within valid range)
        best_contour = max(valid_contours, key=cv2.contourArea)

        # Step 10: Find dart tip position
        tip_x, tip_y = self._find_dart_tip(best_contour)

        # Transform to full coordinates if needed
        transform = getattr(frame, "transform", None)
        if transform is not None:
            tip_x, tip_y = transform.to_full_coords(tip_x, tip_y)

        # Step 11: Validate position (is it on the board?)
        if not self.mapper.is_valid_hit(tip_x, tip_y):
            logger.debug(f"Position ({tip_x:.0f}, {tip_y:.0f}) not on board")
            return None

        # Step 12: Check if too close to last hit
        if self.last_hit:
            distance = np.sqrt(
                (tip_x - self.last_hit.x_px)**2 +
                (tip_y - self.last_hit.y_px)**2
            )
            if distance < 20:  # 20 pixels minimum distance
                logger.debug(f"Too close to last hit ({distance:.1f}px)")
                return None

        # Step 13: Convert to score - WE FOUND A HIT! 🎯
        hit = self.mapper.pixel_to_score(
            tip_x, tip_y,
            frame_id=getattr(frame, 'frame_id', 0),
            timestamp=current_time
        )

        # Step 14: Update state
        self.last_hit = hit
        self.last_hit_time = current_time
        self.looking_for_dart = False
        self.motion_stopped_time = None

        logger.info(f"✓ HIT CONFIRMED: {hit.score} points at ({tip_x:.1f}, {tip_y:.1f})")

        return hit

    def _find_dart_tip(self, contour: np.ndarray) -> Tuple[float, float]:
        """
        Find dart tip - simple and effective.

        Strategy:
        1. Get contour center (moments)
        2. Find sharpest point (likely the tip)
        3. Weight them together

        Args:
            contour: Detected contour

        Returns:
            (x, y) tip position
        """
        # Method 1: Contour center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            # Fallback to bounding box center
            x, y, w, h = cv2.boundingRect(contour)
            return (x + w/2, y + h/2)

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Method 2: Find sharpest point (tip)
        hull = cv2.convexHull(contour, returnPoints=True)

        if len(hull) < 3:
            return (cx, cy)  # Not enough points

        # Find point with smallest angle (sharpest)
        min_angle = np.pi
        tip_x, tip_y = cx, cy

        for i in range(len(hull)):
            p1 = hull[i-1][0]
            p2 = hull[i][0]
            p3 = hull[(i+1) % len(hull)][0]

            # Vectors from p2 to neighbors
            v1 = p1 - p2
            v2 = p3 - p2

            # Calculate angle
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                if angle < min_angle:
                    min_angle = angle
                    tip_x, tip_y = float(p2[0]), float(p2[1])

        # Combine: 30% center, 70% tip (tip is usually more accurate)
        final_x = 0.3 * cx + 0.7 * tip_x
        final_y = 0.3 * cy + 0.7 * tip_y

        return (final_x, final_y)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def reset(self) -> None:
        """Reset detector state."""
        self.motion_detector.reset()
        self.looking_for_dart = False
        self.motion_stopped_time = None
        self.last_motion_time = None
        self.last_hit = None
        self.last_hit_time = None
        self.background_frame = None
        self.quiet_frames.clear()
        logger.info("Detector reset")

    def get_stats(self) -> dict:
        """Get detection statistics."""
        motion_stats = self.motion_detector.get_stats()
        return {
            **motion_stats,
            "looking_for_dart": self.looking_for_dart,
            "last_hit_time": self.last_hit_time,
            "quiet_frames": len(self.quiet_frames),
        }
