import textwrap
from pathlib import Path

import pytest

pytest.importorskip("cv2", exc_type=ImportError)

from src.detection import (
    HitDetectionConfig,
    build_hit_detection_config,
    load_detection_config,
    load_detection_settings,
)


def test_load_detection_config_missing_file(tmp_path: Path):
    """Missing YAML should fall back to defaults without error."""
    missing_path = tmp_path / "no_config.yaml"
    default_config = HitDetectionConfig()

    loaded = load_detection_config(missing_path)

    assert isinstance(loaded, HitDetectionConfig)
    assert loaded.min_contour_area == default_config.min_contour_area
    assert loaded.motion_config.history == default_config.motion_config.history


def test_build_hit_detection_config_applies_overrides(tmp_path: Path):
    """Overrides from YAML should populate motion, detection, and state configs."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(textwrap.dedent("""
        motion_detection:
          history: 10
          min_motion_area: 5
        hit_detection:
          min_contour_area: 42
          enable_subpixel: false
          confirmation_frames: 2
        detection_state:
          cooldown_radius_px: 75
          min_frames_since_hit: 3
          min_pixel_drift_px: 9.0
    """).strip())

    settings = load_detection_settings(config_path)
    config = build_hit_detection_config(settings)

    assert config.motion_config.history == 10
    assert config.motion_config.min_motion_area == 5
    assert config.min_contour_area == 42
    assert config.enable_subpixel is False
    assert config.confirmation_frames == 2
    assert config.state_config.cooldown_radius_px == 75
    assert config.min_frames_since_hit == 3
    assert config.min_pixel_drift_since_hit == 9.0
