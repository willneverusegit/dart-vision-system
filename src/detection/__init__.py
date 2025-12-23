"""
Detection module - motion detection and hit recognition.
"""
from .motion import MotionDetector, MotionConfig
from .hit_detector import HitDetector, HitDetectionConfig, HitCandidate

__all__ = [
    "MotionDetector",
    "MotionConfig",
    "HitDetector",
    "HitDetectionConfig",
    "HitCandidate",
]