"""
FPS limiter for battery saving and performance control.
"""
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FPSLimiter:
    """
    FPS limiter for controlled frame rate.

    Benefits:
    - Reduces CPU usage and heat
    - Extends battery life
    - Prevents overprocessing

    Usage:
        limiter = FPSLimiter(target_fps=15)

        while True:
            # Process frame
            process_frame()

            # Wait to maintain FPS
            limiter.wait()
    """

    def __init__(self, target_fps: float = 30.0):
        """
        Initialize FPS limiter.

        Args:
            target_fps: Target frames per second (0 = unlimited)
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0.0
        self.last_frame_time = time.time()

        # Statistics
        self.actual_fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()

        logger.info(f"FPSLimiter initialized: target={target_fps} FPS")

    def wait(self) -> float:
        """
        Wait to maintain target FPS.

        Returns:
            Actual time since last frame (seconds)
        """
        current_time = time.time()
        elapsed = current_time - self.last_frame_time

        # Calculate sleep time
        if self.frame_time > 0:
            sleep_time = self.frame_time - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)
                current_time = time.time()
                elapsed = current_time - self.last_frame_time

        # Update statistics
        self.last_frame_time = current_time
        self.frame_count += 1

        # Calculate actual FPS (sliding window)
        total_time = current_time - self.start_time
        if total_time > 0:
            self.actual_fps = self.frame_count / total_time

        return elapsed

    def reset(self) -> None:
        """Reset statistics."""
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
        self.actual_fps = 0.0

    def set_target_fps(self, target_fps: float) -> None:
        """
        Update target FPS.

        Args:
            target_fps: New target FPS (0 = unlimited)
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0.0
        logger.info(f"FPS target updated: {target_fps}")

    @property
    def is_meeting_target(self) -> bool:
        """Check if meeting target FPS (within 10%)."""
        if self.target_fps == 0:
            return True

        tolerance = self.target_fps * 0.1
        return abs(self.actual_fps - self.target_fps) <= tolerance

    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            "target_fps": self.target_fps,
            "actual_fps": self.actual_fps,
            "frame_count": self.frame_count,
            "meeting_target": self.is_meeting_target,
        }


class AdaptiveFPSLimiter(FPSLimiter):
    """
    Adaptive FPS limiter that adjusts based on detection state.

    Strategy:
    - IDLE: Low FPS (10-15) for battery saving
    - WATCHING/CONFIRMING: High FPS (25-30) for accuracy
    - COOLDOWN: Medium FPS (15-20)
    """

    def __init__(
            self,
            idle_fps: float = 10.0,
            active_fps: float = 25.0,
            cooldown_fps: float = 15.0
    ):
        """
        Initialize adaptive limiter.

        Args:
            idle_fps: FPS during IDLE state
            active_fps: FPS during WATCHING/CONFIRMING
            cooldown_fps: FPS during COOLDOWN
        """
        super().__init__(target_fps=idle_fps)

        self.idle_fps = idle_fps
        self.active_fps = active_fps
        self.cooldown_fps = cooldown_fps

        self.current_mode = "idle"

        logger.info(
            f"AdaptiveFPSLimiter: "
            f"idle={idle_fps}, active={active_fps}, cooldown={cooldown_fps}"
        )

    def set_mode(self, mode: str) -> None:
        """
        Set FPS mode based on detection state.

        Args:
            mode: "idle", "active", or "cooldown"
        """
        if mode == self.current_mode:
            return

        self.current_mode = mode

        if mode == "idle":
            self.set_target_fps(self.idle_fps)
        elif mode == "active":
            self.set_target_fps(self.active_fps)
        elif mode == "cooldown":
            self.set_target_fps(self.cooldown_fps)
        else:
            logger.warning(f"Unknown mode: {mode}")