"""
Core data types for dart vision system.
Defines contracts between modules to ensure stable interfaces.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray


@dataclass
class Frame:
    """
    Represents a single captured frame with metadata.
    """
    image: NDArray[np.uint8]  # Raw image data (H, W, C) or (H, W)
    timestamp: float  # Unix timestamp
    frame_id: int  # Sequential frame counter
    fps: Optional[float] = None  # Frames per second at capture time

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.image.shape

    @property
    def is_grayscale(self) -> bool:
        return len(self.image.shape) == 2


@dataclass
class ROI:
    """
    Region of Interest - defines the cropped dartboard area.
    Coordinates in original image space.
    """
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int  # ROI width in pixels
    height: int  # ROI height in pixels

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI dimensions must be positive")

    def crop(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Extract ROI from image."""
        return image[self.y:self.y + self.height, self.x:self.x + self.width]

    @property
    def center(self) -> Tuple[int, int]:
        """Return center coordinates (x, y)."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class CalibrationData:
    """
    Calibration parameters for perspective correction and scaling.
    """
    homography_matrix: NDArray[np.float64]  # 3x3 transformation matrix
    board_center: Tuple[float, float]  # (x, y) in warped coordinates
    mm_per_pixel: float  # Scaling factor: millimeters per pixel
    board_radius_px: float  # Outer double ring radius in pixels
    method: str = "manual"  # "charuco", "manual", "aruco"
    timestamp: Optional[float] = None  # When calibration was performed

    def __post_init__(self):
        if self.homography_matrix.shape != (3, 3):
            raise ValueError("Homography must be 3x3 matrix")
        if self.mm_per_pixel <= 0:
            raise ValueError("mm_per_pixel must be positive")


@dataclass
class BoardGeometry:
    """
    Dartboard geometric parameters (official dimensions).
    All measurements in millimeters unless specified.
    """
    # Radii (from center)
    inner_bull_radius: float = 6.35  # Double bull (50 points)
    outer_bull_radius: float = 15.9  # Single bull (25 points)
    triple_inner_radius: float = 99.0  # Inner edge of triple ring
    triple_outer_radius: float = 107.0  # Outer edge of triple ring
    double_inner_radius: float = 162.0  # Inner edge of double ring
    double_outer_radius: float = 170.0  # Outer edge of double ring (board edge)

    # Sector configuration
    num_sectors: int = 20
    sector_angle: float = 18.0  # Degrees per sector
    sector_sequence: Tuple[int, ...] = (20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                                        3, 19, 7, 16, 8, 11, 14, 9, 12, 5)
    reference_angle: float = 0.0  # Angle offset (20 at top = 0°, adjust if needed)


@dataclass
class Hit:
    """
    Represents a detected dart hit with coordinates and scoring.
    """
    # Image coordinates
    x_px: float  # X coordinate in warped board space
    y_px: float  # Y coordinate in warped board space

    # Polar coordinates (relative to board center)
    radius: float  # Distance from center (in pixels or mm)
    angle: float  # Angle in degrees (0° = top, clockwise)

    # Scoring
    sector: Optional[int] = None  # Number hit (1-20)
    multiplier: Optional[int] = None  # 1=Single, 2=Double, 3=Triple, 25=Bull, 50=Double Bull
    score: Optional[int] = None  # Total points (sector * multiplier)

    # Metadata
    confidence: float = 1.0  # Detection confidence (0.0 - 1.0)
    timestamp: Optional[float] = None
    frame_id: Optional[int] = None


@dataclass
class GameState:
    """
    Current state of a dart game.
    """
    game_mode: str = "501"  # "501", "301", "cricket", "around_the_clock"
    players: List[str] = field(default_factory=list)
    current_player_idx: int = 0
    scores: List[int] = field(default_factory=list)  # Remaining points (for 501) or current score
    darts_thrown: int = 0  # Darts in current turn (max 3)
    hits_this_turn: List[Hit] = field(default_factory=list)
    total_hits: List[List[Hit]] = field(default_factory=list)  # History per player

    def __post_init__(self):
        if not self.scores and self.players:
            # Initialize scores based on game mode
            if self.game_mode in ["501", "301"]:
                initial_score = int(self.game_mode)
                self.scores = [initial_score] * len(self.players)
            else:
                self.scores = [0] * len(self.players)

        if not self.total_hits and self.players:
            self.total_hits = [[] for _ in self.players]

    @property
    def current_player(self) -> Optional[str]:
        if 0 <= self.current_player_idx < len(self.players):
            return self.players[self.current_player_idx]
        return None

    @property
    def current_score(self) -> Optional[int]:
        if 0 <= self.current_player_idx < len(self.scores):
            return self.scores[self.current_player_idx]
        return None