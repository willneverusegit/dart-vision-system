"""Tests for stereo calibration and triangulation."""

import numpy as np

from backend.models.camera import Intrinsics
from backend.models.stereo import StereoProfile
from backend.vision.stereo import (
    calibrate_stereo,
    compute_reprojection_error,
    project_to_board_plane,
    triangulate_point,
)


def _make_intrinsics(
    fx: float = 800.0, fy: float = 800.0, cx: float = 320.0, cy: float = 240.0
) -> Intrinsics:
    """Create test intrinsics with zero distortion."""
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0])


def _make_stereo_profile(
    rotation: list[list[float]] | None = None,
    translation: list[float] | None = None,
) -> StereoProfile:
    """Create a stereo profile with known R and T."""
    if rotation is None:
        rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if translation is None:
        translation = [100.0, 0.0, 0.0]
    return StereoProfile(
        camera_left_id="left",
        camera_right_id="right",
        rotation_matrix=rotation,
        translation_vector=translation,
        reprojection_error=0.0,
    )


def _project_point(
    point_3d: np.ndarray, cam_matrix: np.ndarray, rot: np.ndarray, tvec: np.ndarray
) -> tuple[float, float]:
    """Project a 3D world point into a camera image plane."""
    pt_cam = rot @ point_3d.reshape(3, 1) + tvec.reshape(3, 1)
    pt_img = cam_matrix @ pt_cam
    return float(pt_img[0, 0] / pt_img[2, 0]), float(pt_img[1, 0] / pt_img[2, 0])


class TestCalibrateStereo:
    """Tests for calibrate_stereo function."""

    def test_calibrate_stereo_returns_profile(self) -> None:
        """Stereo calibration with synthetic data returns a valid StereoProfile."""
        left_intr = _make_intrinsics()
        right_intr = _make_intrinsics()
        left_K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        right_K = left_K.copy()

        R_true = np.eye(3, dtype=np.float64)
        T_true = np.array([[100.0], [0.0], [0.0]], dtype=np.float64)

        # Generate synthetic charuco-like points on a planar grid
        board_from = __import__("backend.vision.calibration", fromlist=["get_charuco_board"])
        board = board_from.get_charuco_board()
        obj_pts_all = board.getChessboardCorners()

        rng = np.random.default_rng(42)
        left_corners_list = []
        left_ids_list = []
        right_corners_list = []
        right_ids_list = []

        num_frames = 5
        num_corners = min(20, len(obj_pts_all))

        for _ in range(num_frames):
            # Small random rotation for variety
            angle = rng.uniform(-0.1, 0.1)
            R_board = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            T_board = np.array([[0], [0], [500 + rng.uniform(-50, 50)]], dtype=np.float64)

            ids = np.arange(num_corners, dtype=np.int32)
            left_pts = []
            right_pts = []

            for i in range(num_corners):
                pt_world = R_board @ obj_pts_all[i].reshape(3, 1) + T_board
                # Left camera at origin
                pl = left_K @ pt_world
                left_pts.append([float(pl[0, 0] / pl[2, 0]), float(pl[1, 0] / pl[2, 0])])
                # Right camera
                pt_right = R_true @ pt_world + T_true
                pr = right_K @ pt_right
                right_pts.append([float(pr[0, 0] / pr[2, 0]), float(pr[1, 0] / pr[2, 0])])

            left_corners_list.append(np.array(left_pts, dtype=np.float32).reshape(-1, 1, 2))
            left_ids_list.append(ids.reshape(-1, 1))
            right_corners_list.append(np.array(right_pts, dtype=np.float32).reshape(-1, 1, 2))
            right_ids_list.append(ids.reshape(-1, 1))

        result = calibrate_stereo(
            left_corners_list,
            left_ids_list,
            right_corners_list,
            right_ids_list,
            left_intr,
            right_intr,
            (640, 480),
            board,
        )

        assert result is not None
        assert isinstance(result, StereoProfile)
        assert len(result.rotation_matrix) == 3
        assert len(result.translation_vector) == 3
        assert result.reprojection_error >= 0.0

    def test_calibrate_stereo_empty_input(self) -> None:
        """Returns None when no frames are provided."""
        result = calibrate_stereo(
            [],
            [],
            [],
            [],
            _make_intrinsics(),
            _make_intrinsics(),
            (640, 480),
        )
        assert result is None


class TestTriangulatePoint:
    """Tests for triangulate_point function."""

    def test_triangulate_known_point(self) -> None:
        """Triangulating a known 3D point gives back the original within 1mm."""
        intr = _make_intrinsics()
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        T = np.array([[100.0], [0.0], [0.0]], dtype=np.float64)

        # Known 3D point
        pt_3d = np.array([50.0, 30.0, 500.0])

        # Project to left (identity)
        pl = _project_point(pt_3d, K, np.eye(3), np.zeros((3, 1)))
        # Project to right
        pr = _project_point(pt_3d, K, R, T)

        stereo = _make_stereo_profile(rotation=R.tolist(), translation=T.flatten().tolist())

        result = triangulate_point(pl, pr, intr, intr, stereo)

        assert abs(result[0] - pt_3d[0]) < 1.0
        assert abs(result[1] - pt_3d[1]) < 1.0
        assert abs(result[2] - pt_3d[2]) < 1.0


class TestProjectToBoardPlane:
    """Tests for project_to_board_plane function."""

    def test_project_to_board_plane(self) -> None:
        """Projecting (10, 20, 5) to z=0 returns (10, 20)."""
        result = project_to_board_plane((10.0, 20.0, 5.0))
        assert abs(result[0] - 10.0) < 0.01
        assert abs(result[1] - 20.0) < 0.01

    def test_project_already_on_plane(self) -> None:
        """Point already at z=0 is returned unchanged."""
        result = project_to_board_plane((15.0, 25.0, 0.0))
        assert result == (15.0, 25.0)


class TestReprojectionError:
    """Tests for compute_reprojection_error function."""

    def test_reprojection_error_small(self) -> None:
        """Correctly triangulated point has reprojection error < 1 pixel."""
        intr = _make_intrinsics()
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        T = np.array([[100.0], [0.0], [0.0]], dtype=np.float64)

        pt_3d_orig = np.array([50.0, 30.0, 500.0])

        pl = _project_point(pt_3d_orig, K, np.eye(3), np.zeros((3, 1)))
        pr = _project_point(pt_3d_orig, K, R, T)

        stereo = _make_stereo_profile(rotation=R.tolist(), translation=T.flatten().tolist())

        # Triangulate
        pt_3d = triangulate_point(pl, pr, intr, intr, stereo)

        error = compute_reprojection_error(pt_3d, pl, pr, intr, intr, stereo)
        assert error < 1.0
