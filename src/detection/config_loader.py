"""
Utilities to load detection-related configuration from YAML files.

The goal is to keep the tuning workflow in `config/default_config.yaml` so
developers can iteratively adjust hit detection parameters without touching
code. Unknown keys are ignored to keep the loader backwards compatible.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from src.core import load_yaml
from .hit_detector import HitDetectionConfig
from .motion import MotionConfig
from .detection_state import StateConfig

logger = logging.getLogger(__name__)

# Default location for the application-wide settings
DEFAULT_CONFIG_PATH = Path("config/default_config.yaml")


def _apply_overrides(target: Any, overrides: Dict[str, Any]) -> None:
    """
    Apply dictionary overrides to a dataclass-like object.

    Unknown keys are ignored to remain forward compatible with new YAML fields.
    """
    for key, value in overrides.items():
        if hasattr(target, key):
            setattr(target, key, value)
        else:
            logger.debug("Ignoring unknown config key: %s", key)


def load_detection_settings(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load raw configuration dictionary from YAML.

    Args:
        config_path: Optional path to YAML file (defaults to DEFAULT_CONFIG_PATH)

    Returns:
        Dictionary with configuration values (empty dict on failure)
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.info("Detection config not found at %s, using defaults", path)
        return {}

    try:
        return load_yaml(path) or {}
    except Exception as exc:  # YAML/IO errors fall back to safe defaults
        logger.warning("Failed to load detection config from %s: %s", path, exc)
        return {}


def build_hit_detection_config(settings: Optional[Dict[str, Any]] = None) -> HitDetectionConfig:
    """
    Construct HitDetectionConfig (including motion/state configs) from settings.

    Args:
        settings: Raw settings dictionary (e.g., from load_detection_settings)

    Returns:
        Populated HitDetectionConfig instance
    """
    settings = settings or {}
    detection_overrides = settings.get("hit_detection") or settings.get("detection") or {}
    motion_overrides = settings.get("motion_detection") or settings.get("motion") or {}
    state_overrides = settings.get("detection_state") or {}

    motion_config = MotionConfig()
    state_config = StateConfig()
    detection_config = HitDetectionConfig(
        motion_config=motion_config,
        state_config=state_config,
    )

    _apply_overrides(motion_config, motion_overrides)
    _apply_overrides(state_config, state_overrides)
    _apply_overrides(detection_config, detection_overrides)

    # Keep spacing/cooldown in sync when users only set it in one section
    if "min_frames_since_hit" in state_overrides and "min_frames_since_hit" not in detection_overrides:
        detection_config.min_frames_since_hit = state_config.min_frames_since_hit
    if "min_pixel_drift_px" in state_overrides and "min_pixel_drift_since_hit" not in detection_overrides:
        detection_config.min_pixel_drift_since_hit = state_config.min_pixel_drift_px

    return detection_config


def load_detection_config(config_path: Optional[Path] = None) -> HitDetectionConfig:
    """
    Convenience wrapper to load and build a detection config in one call.
    """
    settings = load_detection_settings(config_path)
    return build_hit_detection_config(settings)
