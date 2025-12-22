"""
Unit tests for core module.
"""
import numpy as np
from pathlib import Path
import tempfile
import pytest

from src.core import (
    Frame, ROI, CalibrationData, BoardGeometry, Hit, GameState,
    atomic_write_yaml, load_yaml
)


def test_frame_creation():
    """Test Frame dataclass."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(image=img, timestamp=123.45, frame_id=1, fps=30.0)

    assert frame.shape == (480, 640, 3)
    assert not frame.is_grayscale

    # Test grayscale
    gray_img = np.zeros((480, 640), dtype=np.uint8)
    gray_frame = Frame(image=gray_img, timestamp=123.45, frame_id=2)
    assert gray_frame.is_grayscale


def test_roi():
    """Test ROI cropping."""
    roi = ROI(x=100, y=100, width=200, height=150)
    assert roi.center == (200, 175)

    # Test cropping
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cropped = roi.crop(img)
    assert cropped.shape == (150, 200, 3)

    # Test invalid dimensions
    with pytest.raises(ValueError):
        ROI(x=0, y=0, width=-10, height=100)


def test_calibration_data():
    """Test CalibrationData validation."""
    H = np.eye(3, dtype=np.float64)
    calib = CalibrationData(
        homography_matrix=H,
        board_center=(320.0, 240.0),
        mm_per_pixel=0.5,
        board_radius_px=200.0
    )
    assert calib.method == "manual"

    # Test invalid homography
    with pytest.raises(ValueError):
        CalibrationData(
            homography_matrix=np.zeros((2, 2)),
            board_center=(0, 0),
            mm_per_pixel=1.0,
            board_radius_px=100.0
        )


def test_board_geometry():
    """Test BoardGeometry defaults."""
    board = BoardGeometry()
    assert board.num_sectors == 20
    assert board.sector_angle == 18.0
    assert len(board.sector_sequence) == 20
    assert board.sector_sequence[0] == 20  # Top sector


def test_hit():
    """Test Hit dataclass."""
    hit = Hit(
        x_px=120.5, y_px=100.3,
        radius=50.0, angle=45.0,
        sector=20, multiplier=3, score=60
    )
    assert hit.score == 60
    assert hit.confidence == 1.0


def test_game_state():
    """Test GameState initialization."""
    players = ["Alice", "Bob"]
    game = GameState(game_mode="501", players=players)

    assert game.scores == [501, 501]
    assert game.current_player == "Alice"
    assert game.current_score == 501
    assert len(game.total_hits) == 2


def test_atomic_write_yaml():
    """Test atomic YAML writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_config.yaml"

        data = {
            "camera": {"index": 0},
            "calibration": {"mm_per_pixel": 0.5}
        }

        # Write
        atomic_write_yaml(filepath, data)
        assert filepath.exists()

        # Read back
        loaded = load_yaml(filepath)
        assert loaded["camera"]["index"] == 0
        assert loaded["calibration"]["mm_per_pixel"] == 0.5


def test_load_nonexistent_yaml():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_yaml(Path("nonexistent_file.yaml"))


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running basic tests...")
    test_frame_creation()
    print("✓ Frame tests passed")
    test_roi()
    print("✓ ROI tests passed")
    test_calibration_data()
    print("✓ CalibrationData tests passed")
    test_board_geometry()
    print("✓ BoardGeometry tests passed")
    test_hit()
    print("✓ Hit tests passed")
    test_game_state()
    print("✓ GameState tests passed")
    test_atomic_write_yaml()
    print("✓ Atomic write tests passed")
    print("\n✓ All tests passed!")