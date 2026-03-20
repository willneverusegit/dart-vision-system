from pydantic import BaseModel


class BoardModel(BaseModel):
    """Mathematical model of the dartboard derived from calibration."""

    homography: list[list[float]]  # 3x3 matrix
    board_radius: float = 170.0  # mm, outer double ring
    ring_radii: dict[str, float] = {
        "bull": 6.35,
        "outer_bull": 15.9,
        "triple_inner": 99.0,
        "triple_outer": 107.0,
        "double_inner": 162.0,
        "double_outer": 170.0,
    }
    sector_angles: list[int] = [
        20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
        3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
    ]
