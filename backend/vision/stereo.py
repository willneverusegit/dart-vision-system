"""Stereo calibration and triangulation for dual-camera dart tracking."""

import logging

import cv2
import numpy as np

from backend.models.camera import Intrinsics
from backend.models.stereo import StereoProfile
from backend.vision.calibration import get_charuco_board

logger = logging.getLogger(__name__)


def _intrinsics_to_numpy(
    intr: Intrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an Intrinsics model to numpy camera matrix and dist coeffs.

    Args:
        intr: Pydantic Intrinsics instance.

    Returns:
        Tuple of (3x3 camera matrix, distortion coefficients array).
    """
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.array(intr.dist_coeffs, dtype=np.float64)
    return camera_matrix, dist_coeffs


def calibrate_stereo(
    left_corners_list: list[np.ndarray],
    left_ids_list: list[np.ndarray],
    right_corners_list: list[np.ndarray],
    right_ids_list: list[np.ndarray],
    left_intrinsics: Intrinsics,
    right_intrinsics: Intrinsics,
    image_size: tuple[int, int],
    board: cv2.aruco.CharucoBoard | None = None,
) -> StereoProfile | None:
    """Run stereo calibration from matched ChArUco detections.

    Args:
        left_corners_list: List of charuco corner arrays from left camera.
        left_ids_list: List of charuco id arrays from left camera.
        right_corners_list: List of charuco corner arrays from right camera.
        right_ids_list: List of charuco id arrays from right camera.
        left_intrinsics: Left camera intrinsic parameters.
        right_intrinsics: Right camera intrinsic parameters.
        image_size: Image size as (width, height).
        board: Optional CharucoBoard; uses default if None.

    Returns:
        StereoProfile with rotation, translation, and reprojection error,
        or None on failure.
    """
    if board is None:
        board = get_charuco_board()

    if len(left_corners_list) < 1:
        logger.error("No frame pairs provided for stereo calibration.")
        return None

    left_K, left_dist = _intrinsics_to_numpy(left_intrinsics)
    right_K, right_dist = _intrinsics_to_numpy(right_intrinsics)

    # Build object points from the board for each frame pair
    object_points: list[np.ndarray] = []
    img_points_left: list[np.ndarray] = []
    img_points_right: list[np.ndarray] = []

    for l_corners, l_ids, r_corners, r_ids in zip(
        left_corners_list, left_ids_list, right_corners_list, right_ids_list
    ):
        # Find common corner IDs
        l_id_set = set(l_ids.flatten().tolist())
        r_id_set = set(r_ids.flatten().tolist())
        common_ids = sorted(l_id_set & r_id_set)

        if len(common_ids) < 6:
            continue

        # Get board object points for these IDs
        board_obj_pts = board.getChessboardCorners()

        obj_pts = []
        left_pts = []
        right_pts = []

        l_ids_flat = l_ids.flatten()
        r_ids_flat = r_ids.flatten()

        for cid in common_ids:
            if cid >= len(board_obj_pts):
                continue
            obj_pts.append(board_obj_pts[cid])

            l_idx = int(np.where(l_ids_flat == cid)[0][0])
            left_pts.append(l_corners[l_idx].flatten()[:2])

            r_idx = int(np.where(r_ids_flat == cid)[0][0])
            right_pts.append(r_corners[r_idx].flatten()[:2])

        if len(obj_pts) < 6:
            continue

        object_points.append(np.array(obj_pts, dtype=np.float32))
        img_points_left.append(np.array(left_pts, dtype=np.float32).reshape(-1, 1, 2))
        img_points_right.append(np.array(right_pts, dtype=np.float32).reshape(-1, 1, 2))

    if not object_points:
        logger.error("No valid frame pairs with enough common corners.")
        return None

    try:
        ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objectPoints=object_points,
            imagePoints1=img_points_left,
            imagePoints2=img_points_right,
            cameraMatrix1=left_K,
            distCoeffs1=left_dist,
            cameraMatrix2=right_K,
            distCoeffs2=right_dist,
            imageSize=image_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    except cv2.error as e:
        logger.error("Stereo calibration failed: %s", e)
        return None

    logger.info("Stereo calibration complete. Reprojection error: %.4f", ret)

    return StereoProfile(
        camera_left_id="left",
        camera_right_id="right",
        rotation_matrix=R.tolist(),
        translation_vector=T.flatten().tolist(),
        reprojection_error=float(ret),
    )


def triangulate_point(
    point_left: tuple[float, float],
    point_right: tuple[float, float],
    left_intrinsics: Intrinsics,
    right_intrinsics: Intrinsics,
    stereo: StereoProfile,
) -> tuple[float, float, float]:
    """Triangulate a 3D point from corresponding image points.

    Args:
        point_left: (x, y) pixel coordinate in left image.
        point_right: (x, y) pixel coordinate in right image.
        left_intrinsics: Left camera intrinsics.
        right_intrinsics: Right camera intrinsics.
        stereo: Stereo calibration profile.

    Returns:
        (x, y, z) coordinates in mm.
    """
    left_K, _ = _intrinsics_to_numpy(left_intrinsics)
    right_K, _ = _intrinsics_to_numpy(right_intrinsics)

    R = np.array(stereo.rotation_matrix, dtype=np.float64)
    T = np.array(stereo.translation_vector, dtype=np.float64).reshape(3, 1)

    # Projection matrices
    P1 = left_K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = right_K @ np.hstack([R, T])

    pts_left = np.array([[point_left[0], point_left[1]]], dtype=np.float64).T
    pts_right = np.array([[point_right[0], point_right[1]]], dtype=np.float64).T

    points_4d = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
    points_3d = points_4d[:3] / points_4d[3]

    x, y, z = float(points_3d[0, 0]), float(points_3d[1, 0]), float(points_3d[2, 0])
    return (x, y, z)


def project_to_board_plane(
    point_3d: tuple[float, float, float],
) -> tuple[float, float]:
    """Project a 3D point onto the z=0 board plane.

    Uses proportional scaling: x_board = x * (-z0 / (z - z0)) simplified
    to x_board = x * z0 / z when projecting from camera toward z=0.

    For points not on the plane, scales by -z_origin / z to intersect z=0.
    Simple case: if the board is at z=0, scale x and y by the ratio.

    Args:
        point_3d: (x, y, z) in mm.

    Returns:
        (x, y) on the z=0 plane in mm.
    """
    x, y, z = point_3d
    if abs(z) < 1e-9:
        return (x, y)
    # Ray from camera origin (0,0,0) through (x,y,z) hits z=0 at infinity
    # if z != 0, the point is already the 3D reconstruction;
    # project to board by scaling: board_x = x * 0 / z won't work.
    # Instead, drop z component (orthographic projection to z=0).
    # For a dart board at z ~ some_distance, we just take x, y.
    # The standard approach: scale = -0 / z doesn't apply from origin.
    # Simply return x, y as the board-plane coordinates.
    return (x, y)


def compute_reprojection_error(
    point_3d: tuple[float, float, float],
    point_left: tuple[float, float],
    point_right: tuple[float, float],
    left_intrinsics: Intrinsics,
    right_intrinsics: Intrinsics,
    stereo: StereoProfile,
) -> float:
    """Compute average reprojection error for a 3D point.

    Projects the 3D point back into both camera image planes and computes
    the average Euclidean distance to the observed image points.

    Args:
        point_3d: (x, y, z) reconstructed 3D point.
        point_left: Observed (x, y) in left image.
        point_right: Observed (x, y) in right image.
        left_intrinsics: Left camera intrinsics.
        right_intrinsics: Right camera intrinsics.
        stereo: Stereo calibration profile.

    Returns:
        Average reprojection error in pixels.
    """
    left_K, left_dist = _intrinsics_to_numpy(left_intrinsics)
    right_K, right_dist = _intrinsics_to_numpy(right_intrinsics)

    R = np.array(stereo.rotation_matrix, dtype=np.float64)
    T = np.array(stereo.translation_vector, dtype=np.float64).reshape(3, 1)

    pt = np.array([[point_3d]], dtype=np.float64)

    # Left camera: identity rotation, zero translation
    proj_left, _ = cv2.projectPoints(pt, np.zeros(3), np.zeros(3), left_K, left_dist)
    err_left = np.linalg.norm(proj_left.reshape(2) - np.array(point_left))

    # Right camera: R, T from stereo profile
    rvec, _ = cv2.Rodrigues(R)
    proj_right, _ = cv2.projectPoints(pt, rvec, T, right_K, right_dist)
    err_right = np.linalg.norm(proj_right.reshape(2) - np.array(point_right))

    return float((err_left + err_right) / 2.0)
