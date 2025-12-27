"""
Detection module - motion detection and hit recognition.
"""
from .motion import MotionDetector, MotionConfig
from .detection_state import (
    DetectionState,
    DetectionStateMachine,
    StateConfig,
)
from .hit_detector import (
    EnhancedHitDetector as HitDetector,  # Use enhanced version
    HitDetectionConfig,
)
from .simple_hit_detector import SimpleHitDetector

__all__ = [
    "MotionDetector",
    "MotionConfig",
    "DetectionState",
    "DetectionStateMachine",
    "StateConfig",
    "HitDetector",
    "HitDetectionConfig",
    "SimpleHitDetector",  # New simplified detector
]