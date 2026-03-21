"""Tests for board fitting and homography."""

import numpy as np

from backend.vision.board_fitting import (
    compute_homography_from_circle,
    compute_homography_from_points,
    pixel_to_board,
    warp_to_topdown,
)


def test_compute_homography_from_points():
    """4-point homography computes a valid 3x3 matrix."""
    src = [(100, 100), (400, 100), (400, 400), (100, 400)]
    dst = [(0, 0), (500, 0), (500, 500), (0, 500)]
    H = compute_homography_from_points(src, dst)
    assert H is not None
    assert H.shape == (3, 3)


def test_compute_homography_too_few_points():
    """Homography requires at least 4 points."""
    src = [(100, 100), (400, 100)]
    dst = [(0, 0), (500, 0)]
    H = compute_homography_from_points(src, dst)
    assert H is None


def test_compute_homography_from_circle():
    """Circle-based homography maps bounding box to square."""
    H = compute_homography_from_circle(center=(320, 240), radius=150, output_size=500)
    assert H is not None
    assert H.shape == (3, 3)


def test_warp_to_topdown():
    """Warp produces correct output dimensions."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    H = compute_homography_from_circle(center=(320, 240), radius=150, output_size=400)
    result = warp_to_topdown(frame, H, output_size=400)
    assert result.shape == (400, 400, 3)


def test_pixel_to_board_center():
    """Center pixel maps to (0, 0) board coordinates."""
    H = np.eye(3)  # identity homography
    x, y = pixel_to_board((250, 250), H, output_size=500)
    assert abs(x) < 1.0
    assert abs(y) < 1.0


def test_pixel_to_board_edge():
    """Edge pixel maps to board radius distance."""
    H = np.eye(3)  # identity
    x, y = pixel_to_board((500, 250), H, output_size=500)
    # Should be at board_radius_mm (170) on x-axis
    assert abs(x - 170.0) < 1.0
    assert abs(y) < 1.0
