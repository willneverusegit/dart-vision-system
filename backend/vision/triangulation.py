"""Multi-camera triangulation with outlier detection.

Extends stereo triangulation to support 3 cameras by triangulating
all camera pairs and using median filtering for robust 3D estimation.
"""

import itertools
import logging

import numpy as np

from backend.models.camera import Intrinsics
from backend.models.stereo import StereoProfile
from backend.vision.stereo import triangulate_point

logger = logging.getLogger(__name__)


def multi_triangulate(
    image_points: dict[str, tuple[float, float]],
    intrinsics: dict[str, Intrinsics],
    stereo_profiles: dict[tuple[str, str], StereoProfile],
    max_deviation_mm: float = 5.0,
) -> tuple[float, float, float] | None:
    """Triangulate a 3D point from 2-3 camera observations.

    For each valid camera pair with a stereo profile, triangulates a 3D
    point. If 3 cameras yield 3 estimates, uses component-wise median
    and flags outliers that deviate more than max_deviation_mm.

    Args:
        image_points: Mapping of camera_id to (x, y) pixel coordinates.
        intrinsics: Mapping of camera_id to Intrinsics.
        stereo_profiles: Mapping of (left_id, right_id) to StereoProfile.
            Key order matters — (A, B) and (B, A) are different profiles.
        max_deviation_mm: Maximum allowed deviation from median per axis
            before a triangulation is considered an outlier.

    Returns:
        Median (x, y, z) in mm, or None if no valid pair could triangulate.
    """
    camera_ids = list(image_points.keys())
    if len(camera_ids) < 2:
        logger.error("Need at least 2 cameras, got %d", len(camera_ids))
        return None

    estimates: list[tuple[float, float, float]] = []
    pair_labels: list[str] = []

    for cam_a, cam_b in itertools.combinations(camera_ids, 2):
        # Try both orderings for the stereo profile
        profile = stereo_profiles.get((cam_a, cam_b))
        if profile is not None:
            left_id, right_id = cam_a, cam_b
        else:
            profile = stereo_profiles.get((cam_b, cam_a))
            if profile is not None:
                left_id, right_id = cam_b, cam_a
            else:
                logger.debug("No stereo profile for pair (%s, %s)", cam_a, cam_b)
                continue

        left_intr = intrinsics.get(left_id)
        right_intr = intrinsics.get(right_id)
        if left_intr is None or right_intr is None:
            continue

        try:
            pt = triangulate_point(
                point_left=image_points[left_id],
                point_right=image_points[right_id],
                left_intrinsics=left_intr,
                right_intrinsics=right_intr,
                stereo=profile,
            )
            estimates.append(pt)
            pair_labels.append(f"{left_id}-{right_id}")
        except Exception:
            logger.warning(
                "Triangulation failed for pair (%s, %s)",
                left_id,
                right_id,
                exc_info=True,
            )

    if not estimates:
        logger.error("No camera pair produced a valid triangulation.")
        return None

    if len(estimates) == 1:
        return estimates[0]

    # Compute component-wise median
    arr = np.array(estimates, dtype=np.float64)
    median = np.median(arr, axis=0)

    # Outlier detection
    deviations = np.abs(arr - median)
    max_dev_per_estimate = np.max(deviations, axis=1)

    outlier_indices = np.where(max_dev_per_estimate > max_deviation_mm)[0]
    if len(outlier_indices) > 0:
        for idx in outlier_indices:
            logger.warning(
                "Outlier from pair %s: (%.2f, %.2f, %.2f) — deviation %.2f mm from median",
                pair_labels[idx],
                estimates[idx][0],
                estimates[idx][1],
                estimates[idx][2],
                max_dev_per_estimate[idx],
            )

    # If majority are outliers, still return median but log warning
    if len(outlier_indices) > len(estimates) / 2:
        logger.warning(
            "Majority of triangulations are outliers (%d/%d). Result may be unreliable.",
            len(outlier_indices),
            len(estimates),
        )

    # Filter out outliers and re-compute median if we have enough inliers
    inlier_mask = max_dev_per_estimate <= max_deviation_mm
    inliers = arr[inlier_mask]

    if len(inliers) >= 1:
        result = np.median(inliers, axis=0)
    else:
        # All outliers — fall back to full median
        result = median

    return (float(result[0]), float(result[1]), float(result[2]))
