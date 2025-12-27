"""
Detection module - motion detection and hit recognition.
"""
from .motion import MotionDetector, MotionConfig
from .detection_state import (
    DetectionState,
    DetectionStateMachine,
    StateConfig,
)
from .config_loader import (
    build_hit_detection_config,
    load_detection_config,
    load_detection_settings,
)
from .hit_detector import (
    EnhancedHitDetector as HitDetector,  # Use enhanced version
    HitDetectionConfig,
)

__all__ = [
    "MotionDetector",
    "MotionConfig",
    "DetectionState",
    "DetectionStateMachine",
    "StateConfig",
    "HitDetector",
    "HitDetectionConfig",
    "build_hit_detection_config",
    "load_detection_config",
    "load_detection_settings",
]
