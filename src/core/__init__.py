"""
Core module - shared data types, utilities, and configuration.
"""
from .types import (
    Frame,
    ROI,
    CalibrationData,
    BoardGeometry,
    Hit,
    GameState,
)
from .io_utils import (
    atomic_write_yaml,
    load_yaml,
    ensure_config_dir,
    backup_config,
)
from .roi_utils import (
    ROIExtractor,
    PreprocessingPipeline,
)
from .config_loader import Config
from .fps_limiter import (  # ← NEW
    FPSLimiter,
    AdaptiveFPSLimiter,
)
from .performance_monitor import (  # ← NEW
    PerformanceMonitor,
    PerformanceProfiler,
)

__all__ = [
    # Types
    "Frame",
    "ROI",
    "CalibrationData",
    "BoardGeometry",
    "Hit",
    "GameState",
    # I/O
    "atomic_write_yaml",
    "load_yaml",
    "ensure_config_dir",
    "backup_config",
    # ROI
    "ROIExtractor",
    "PreprocessingPipeline",
    # Config
    "Config",
    # Performance
    "FPSLimiter",
    "AdaptiveFPSLimiter",
    "PerformanceMonitor",
    "PerformanceProfiler",
]