"""Tests for multi-camera triangulation."""

from unittest.mock import patch

import numpy as np

from backend.models.camera import Intrinsics
from backend.models.stereo import StereoProfile
from backend.vision.triangulation import multi_triangulate


def _make_intrinsics(fx=500, fy=500, cx=320, cy=240):
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, dist_coeffs=[0, 0, 0, 0, 0])


def _make_stereo_profile():
    return StereoProfile(
        camera_left_id="left",
        camera_right_id="right",
        rotation_matrix=np.eye(3).tolist(),
        translation_vector=[100.0, 0.0, 0.0],
        reprojection_error=0.1,
    )


class TestMultiTriangulate:
    def test_single_camera_returns_none(self):
        """Need at least 2 cameras."""
        result = multi_triangulate(
            image_points={"cam0": (320, 240)},
            intrinsics={"cam0": _make_intrinsics()},
            stereo_profiles={},
        )
        assert result is None

    def test_no_stereo_profile_returns_none(self):
        """Two cameras but no stereo profile."""
        result = multi_triangulate(
            image_points={"cam0": (320, 240), "cam1": (300, 240)},
            intrinsics={"cam0": _make_intrinsics(), "cam1": _make_intrinsics()},
            stereo_profiles={},
        )
        assert result is None

    @patch("backend.vision.triangulation.triangulate_point")
    def test_two_cameras_single_estimate(self, mock_tri):
        """Two cameras with valid profile returns single estimate."""
        mock_tri.return_value = (10.0, 20.0, 300.0)

        result = multi_triangulate(
            image_points={"A": (320, 240), "B": (300, 240)},
            intrinsics={"A": _make_intrinsics(), "B": _make_intrinsics()},
            stereo_profiles={("A", "B"): _make_stereo_profile()},
        )
        assert result == (10.0, 20.0, 300.0)
        mock_tri.assert_called_once()

    @patch("backend.vision.triangulation.triangulate_point")
    def test_three_cameras_median(self, mock_tri):
        """Three cameras yield 3 pairs, median is used."""
        mock_tri.side_effect = [
            (10.0, 20.0, 300.0),  # A-B
            (11.0, 21.0, 301.0),  # A-C
            (10.5, 20.5, 300.5),  # B-C
        ]

        result = multi_triangulate(
            image_points={
                "A": (320, 240),
                "B": (300, 240),
                "C": (310, 235),
            },
            intrinsics={
                "A": _make_intrinsics(),
                "B": _make_intrinsics(),
                "C": _make_intrinsics(),
            },
            stereo_profiles={
                ("A", "B"): _make_stereo_profile(),
                ("A", "C"): _make_stereo_profile(),
                ("B", "C"): _make_stereo_profile(),
            },
        )
        assert result is not None
        x, y, z = result
        assert 10.0 <= x <= 11.0
        assert 20.0 <= y <= 21.0
        assert 300.0 <= z <= 301.0

    @patch("backend.vision.triangulation.triangulate_point")
    def test_outlier_detection(self, mock_tri):
        """One outlier is detected and excluded from median."""
        mock_tri.side_effect = [
            (10.0, 20.0, 300.0),  # A-B: good
            (100.0, 200.0, 500.0),  # A-C: outlier
            (10.2, 20.1, 300.1),  # B-C: good
        ]

        result = multi_triangulate(
            image_points={"A": (320, 240), "B": (300, 240), "C": (310, 235)},
            intrinsics={
                "A": _make_intrinsics(),
                "B": _make_intrinsics(),
                "C": _make_intrinsics(),
            },
            stereo_profiles={
                ("A", "B"): _make_stereo_profile(),
                ("A", "C"): _make_stereo_profile(),
                ("B", "C"): _make_stereo_profile(),
            },
            max_deviation_mm=5.0,
        )
        assert result is not None
        x, y, z = result
        # Should be close to the two good estimates, not the outlier
        assert 9.0 <= x <= 11.0
        assert 19.0 <= y <= 21.0

    @patch("backend.vision.triangulation.triangulate_point")
    def test_reversed_profile_key(self, mock_tri):
        """Profile stored as (B, A) still works for pair (A, B)."""
        mock_tri.return_value = (5.0, 10.0, 200.0)

        result = multi_triangulate(
            image_points={"A": (320, 240), "B": (300, 240)},
            intrinsics={"A": _make_intrinsics(), "B": _make_intrinsics()},
            stereo_profiles={("B", "A"): _make_stereo_profile()},
        )
        assert result == (5.0, 10.0, 200.0)

    @patch("backend.vision.triangulation.triangulate_point")
    def test_partial_failure_still_returns(self, mock_tri):
        """If one pair fails, remaining still produce result."""
        mock_tri.side_effect = [
            Exception("calibration error"),  # A-B fails
            (10.0, 20.0, 300.0),  # A-C works
            (10.1, 20.1, 300.1),  # B-C works
        ]

        result = multi_triangulate(
            image_points={"A": (320, 240), "B": (300, 240), "C": (310, 235)},
            intrinsics={
                "A": _make_intrinsics(),
                "B": _make_intrinsics(),
                "C": _make_intrinsics(),
            },
            stereo_profiles={
                ("A", "B"): _make_stereo_profile(),
                ("A", "C"): _make_stereo_profile(),
                ("B", "C"): _make_stereo_profile(),
            },
        )
        assert result is not None
