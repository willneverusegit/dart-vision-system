"""
Detection state machine for intelligent hit recognition.

State flow based on research:
- IDLE: Waiting for motion
- WATCHING: Motion detected, tracking movement (hand/dart)
- CONFIRMING: Motion stopped, searching for new static object
- COOLDOWN: Hit confirmed, ignore area temporarily

This prevents false positives from:
- Hand movement during throw
- Camera shake
- Shadows and lighting changes
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


class DetectionState(Enum):
    """Detection states for state machine."""
    IDLE = "idle"  # No motion, waiting
    WATCHING = "watching"  # Motion detected, tracking throw
    CONFIRMING = "confirming"  # Motion stopped, searching for dart
    COOLDOWN = "cooldown"  # Hit confirmed, cooldown period


@dataclass
class StateConfig:
    """Configuration for state machine."""
    # Watching state
    watching_duration_sec: float = 1.0  # Max duration in watching state
    large_blob_threshold_px: int = 500  # Hand/arm size threshold

    # Confirming state
    confirming_duration_sec: float = 2.0  # How long to search after motion stops
    confirmation_frames: int = 5  # Frames needed to confirm

    # Cooldown state
    cooldown_duration_sec: float = 1.0  # Cooldown after confirmed hit
    cooldown_radius_px: float = 100.0  # Ignore hits within this radius


class DetectionStateMachine:
    """
    State machine for intelligent dart detection.

    Research-based approach:
    1. IDLE → WATCHING: Significant motion detected
    2. WATCHING → CONFIRMING: Motion subsides
    3. CONFIRMING → IDLE/COOLDOWN: Hit confirmed or timeout
    4. COOLDOWN → IDLE: Cooldown timer expires

    This prevents false positives from hand movement and shadows.
    """

    def __init__(self, config: Optional[StateConfig] = None):
        """
        Initialize state machine.

        Args:
            config: State configuration
        """
        self.config = config or StateConfig()

        # Current state
        self.state = DetectionState.IDLE
        self.state_start_time = time.time()

        # State-specific data
        self.last_hit_position: Optional[tuple] = None
        self.last_hit_time: Optional[float] = None

        # Statistics
        self.state_transitions = 0

        logger.info(
            f"StateMachine initialized: "
            f"watching={self.config.watching_duration_sec}s, "
            f"confirming={self.config.confirming_duration_sec}s, "
            f"cooldown={self.config.cooldown_duration_sec}s"
        )

    def update(
            self,
            has_motion: bool,
            motion_pixels: int = 0,
            confirmed_hit: Optional[tuple] = None
    ) -> DetectionState:
        """
        Update state machine.

        Args:
            has_motion: Whether motion is currently detected
            motion_pixels: Number of motion pixels (for blob size check)
            confirmed_hit: (x, y) position if hit confirmed in current frame

        Returns:
            New state
        """
        current_time = time.time()
        time_in_state = current_time - self.state_start_time

        old_state = self.state

        # State transitions
        if self.state == DetectionState.IDLE:
            if has_motion:
                self._transition_to(DetectionState.WATCHING)

        elif self.state == DetectionState.WATCHING:
            # Check for timeout
            if time_in_state > self.config.watching_duration_sec:
                self._transition_to(DetectionState.CONFIRMING)

            # Check if motion subsided (transition early)
            elif not has_motion:
                self._transition_to(DetectionState.CONFIRMING)

            # Check for very large blob (likely hand, not dart)
            elif motion_pixels > self.config.large_blob_threshold_px:
                logger.debug(f"Large blob detected ({motion_pixels}px), likely hand")
                # Stay in WATCHING, don't transition yet

        elif self.state == DetectionState.CONFIRMING:
            # Hit confirmed
            if confirmed_hit:
                self.last_hit_position = confirmed_hit
                self.last_hit_time = current_time
                self._transition_to(DetectionState.COOLDOWN)

            # Timeout without confirmation
            elif time_in_state > self.config.confirming_duration_sec:
                logger.debug("Confirming timeout, no hit found")
                self._transition_to(DetectionState.IDLE)

        elif self.state == DetectionState.COOLDOWN:
            # Cooldown expired
            if time_in_state > self.config.cooldown_duration_sec:
                self._transition_to(DetectionState.IDLE)

        return self.state

    def _transition_to(self, new_state: DetectionState) -> None:
        """Transition to new state."""
        logger.debug(f"State transition: {self.state.value} → {new_state.value}")
        self.state = new_state
        self.state_start_time = time.time()
        self.state_transitions += 1

    def should_process_detection(self) -> bool:
        """
        Check if detection processing should run.

        Returns:
            True if in CONFIRMING state (searching for dart)
        """
        return self.state == DetectionState.CONFIRMING

    def should_ignore_motion(self) -> bool:
        """
        Check if motion should be ignored (hand filtering).

        Returns:
            True if in WATCHING state (likely hand movement)
        """
        return self.state == DetectionState.WATCHING

    def is_in_cooldown_zone(self, x: float, y: float) -> bool:
        """
        Check if position is in cooldown zone (near recent hit).

        Args:
            x, y: Position to check

        Returns:
            True if within cooldown radius of last hit
        """
        if self.state != DetectionState.COOLDOWN or not self.last_hit_position:
            return False

        last_x, last_y = self.last_hit_position
        distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5

        return distance < self.config.cooldown_radius_px

    def reset(self) -> None:
        """Reset state machine."""
        self.state = DetectionState.IDLE
        self.state_start_time = time.time()
        self.last_hit_position = None
        self.last_hit_time = None
        logger.info("State machine reset")

    def get_stats(self) -> dict:
        """Get state machine statistics."""
        return {
            "current_state": self.state.value,
            "state_transitions": self.state_transitions,
            "time_in_state": time.time() - self.state_start_time,
            "last_hit_time": self.last_hit_time,
        }