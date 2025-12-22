"""
Core module - shared types and utilities.
"""
from .types import (
    Frame,
    ROI,
    CalibrationData,
    BoardGeometry,
    Hit,
    GameState
)
from .io_utils import (
    atomic_write_yaml,
    load_yaml,
    ensure_config_dir,
    backup_config
)

__all__ = [
    # Data types
    "Frame",
    "ROI",
    "CalibrationData",
    "BoardGeometry",
    "Hit",
    "GameState",
    # I/O utilities
    "atomic_write_yaml",
    "load_yaml",
    "ensure_config_dir",
    "backup_config",
]