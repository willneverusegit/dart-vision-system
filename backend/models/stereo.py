from pydantic import BaseModel


class StereoProfile(BaseModel):
    """Extrinsic calibration between two cameras."""

    camera_left_id: str
    camera_right_id: str
    rotation_matrix: list[list[float]]  # 3x3
    translation_vector: list[float]  # 3x1
    reprojection_error: float = 0.0
