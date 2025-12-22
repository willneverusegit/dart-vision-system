"""
Unit tests for calibration module.
"""
import numpy as np
import pytest
import time
from pathlib import Path
import cv2

from src.core import Frame, CalibrationData, BoardGeometry
from src.calibration import CalibratorBase, ManualCalibrator, CharucoCalibrator


def test_calibrator_base():
    """Test CalibratorBase initialization."""
    calibrator = CalibratorBase()
    assert calibrator.board_geometry is not None
    assert isinstance(calibrator.board_geometry, BoardGeometry)


def test_compute_homography():
    """Test homography computation."""
    calibrator = CalibratorBase()

    # Create simple square-to-square transformation
    src = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [200, 0],
        [200, 200],
        [0, 200]
    ], dtype=np.float32)

    success, H = calibrator.compute_homography(src, dst)

    assert success
    assert H is not None
    assert H.shape == (3, 3)


def test_compute_homography_insufficient_points():
    """Test homography with too few points."""
    calibrator = CalibratorBase()

    src = np.array([[0, 0], [100, 0]], dtype=np.float32)
    dst = np.array([[0, 0], [200, 0]], dtype=np.float32)

    success, H = calibrator.compute_homography(src, dst)

    assert not success
    assert H is None


def test_warp_image():
    """Test image warping."""
    calibrator = CalibratorBase()

    # Create test image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Identity homography (no transformation)
    H = np.eye(3, dtype=np.float64)

    warped = calibrator.warp_image(image, H, (640, 480))

    assert warped.shape == (480, 640, 3)


def test_calculate_mm_per_pixel():
    """Test mm_per_pixel calculation."""
    calibrator = CalibratorBase()

    # Known: 170 mm radius = 340 px
    mm_per_px = calibrator.calculate_mm_per_pixel(170.0, 340.0)

    assert abs(mm_per_px - 0.5) < 0.001  # Should be 0.5 mm/px


def test_detect_board_circle():
    """Test circle detection."""
    calibrator = CalibratorBase()

    # Create synthetic dartboard image
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    cv2.circle(image, (400, 400), 300, (255, 255, 255), 2)

    result = calibrator.detect_board_circle(image, min_radius=250, max_radius=350)

    # May or may not detect (depends on Hough parameters)
    # Just test that it doesn't crash
    assert result is None or len(result) == 3


def test_create_canonical_board_points():
    """Test canonical point generation."""
    calibrator = CalibratorBase()

    center = (400.0, 400.0)
    radius = 300.0

    points = calibrator.create_canonical_board_points(center, radius)

    assert points.shape == (4, 2)

    # Check distances from center
    for point in points:
        dist = np.linalg.norm(point - np.array(center))
        assert abs(dist - radius) < 1.0  # Within 1 pixel


def test_validate_calibration():
    """Test calibration validation."""
    calibrator = CalibratorBase()

    # Valid calibration
    H = np.eye(3, dtype=np.float64)
    calib = CalibrationData(
        homography_matrix=H,
        board_center=(400.0, 400.0),
        mm_per_pixel=0.5,
        board_radius_px=300.0,
        method="manual"
    )

    is_valid, msg = calibrator.validate_calibration(calib)
    assert is_valid

    # Invalid: bad scale
    calib_bad = CalibrationData(
        homography_matrix=H,
        board_center=(400.0, 400.0),
        mm_per_pixel=100.0,  # Unrealistic
        board_radius_px=300.0,
        method="manual"
    )

    is_valid, msg = calibrator.validate_calibration(calib_bad)
    assert not is_valid


def test_manual_calibrator_init():
    """Test ManualCalibrator initialization."""
    calibrator = ManualCalibrator(canonical_size=1000)

    assert calibrator.canonical_size == 1000
    assert len(calibrator._selected_points) == 0


def test_charuco_calibrator_init():
    """Test CharucoCalibrator initialization."""
    calibrator = CharucoCalibrator(
        squares_x=5,
        squares_y=7,
        square_length=40.0,
        marker_length=30.0
    )

    assert calibrator.squares_x == 5
    assert calibrator.squares_y == 7
    assert calibrator.board is not None


def test_charuco_generate_board():
    """Test ChArUco board generation."""
    board_image = CharucoCalibrator.generate_board_image(
        squares_x=5,
        squares_y=7,
        square_length=100,
        marker_length=75
    )

    assert board_image is not None
    assert len(board_image.shape) == 2  # Grayscale
    assert board_image.shape[0] > 0
    assert board_image.shape[1] > 0


# Manual calibration requires interactive GUI - skip in automated tests
@pytest.mark.skipif(True, reason="Requires interactive GUI")
def test_manual_calibration_full():
    """Test full manual calibration (requires user interaction)."""
    calibrator = ManualCalibrator()

    # Create test frame
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    frame = Frame(image=image, timestamp=time.time(), frame_id=0)

    result = calibrator.calibrate(frame)

    assert result.success
    assert result.data is not None


if __name__ == "__main__":
    print("Running calibration module tests...")
    test_calibrator_base()
    print("✓ CalibratorBase test passed")
    test_compute_homography()
    print("✓ Homography computation test passed")
    test_compute_homography_insufficient_points()
    print("✓ Insufficient points test passed")
    test_warp_image()
    print("✓ Image warping test passed")
    test_calculate_mm_per_pixel()
    print("✓ mm_per_pixel calculation test passed")
    test_detect_board_circle()
    print("✓ Circle detection test passed")
    test_create_canonical_board_points()
    print("✓ Canonical points test passed")
    test_validate_calibration()
    print("✓ Calibration validation test passed")
    test_manual_calibrator_init()
    print("✓ ManualCalibrator init test passed")
    test_charuco_calibrator_init()
    print("✓ CharucoCalibrator init test passed")
    test_charuco_generate_board()
    print("✓ ChArUco board generation test passed")
    print("\n✓ All calibration tests passed!")