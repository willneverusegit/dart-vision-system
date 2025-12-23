"""
Configuration loader with validation and defaults.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from .io_utils import load_yaml

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration container with research-optimized defaults.
    """

    # Default values based on research
    DEFAULTS = {
        # Performance (ROI-First strategy)
        "performance": {
            "enable_roi": True,
            "roi_margin_px": 50,
            "enable_downscale": True,
            "target_width": 640,
            "enable_fps_limit": False,
            "target_fps": 15,
        },

        # Motion Detection (research-optimized)
        "motion": {
            "history": 150,  # 100-150 recommended
            "var_threshold": 20.0,  # 16-24 recommended
            "learning_rate": 0.005,  # 0.001-0.01 recommended
            "detect_shadows": False,
            "min_motion_area": 50,
            "enable_morphology": True,
            "morph_kernel_size": 3,  # Always 3x3
            "enable_clahe": False,
            "clahe_clip_limit": 2.5,  # 2.0-3.0 recommended
            "clahe_tile_grid_size": [8, 8],
        },

        # Hit Detection (temporal confirmation)
        "detection": {
            "min_contour_area": 10,
            "max_contour_area": 300,
            "confirmation_frames": 5,  # 2-3 recommended minimum
            "position_tolerance_px": 10.0,
            "confirmation_timeout_sec": 3.0,
        },
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Load configuration from file or use defaults.

        Args:
            config_path: Path to config YAML (None = use defaults)
        """
        self.data = self.DEFAULTS.copy()

        if config_path and config_path.exists():
            try:
                user_config = load_yaml(config_path)
                self._merge_config(user_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        else:
            logger.info("Using default configuration")

    def _merge_config(self, user_config: Dict[str, Any]) -> None:
        """Merge user config with defaults."""
        for section, values in user_config.items():
            if section in self.data and isinstance(values, dict):
                self.data[section].update(values)
            else:
                self.data[section] = values

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self.data.get(section, {}).get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section."""
        return self.data.get(section, {})