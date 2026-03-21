"""Tests for board_model scoring: field mapping and score calculation."""

import numpy as np
import pytest

from backend.models.board import BoardModel
from backend.scoring.board_model import board_to_polar, get_score, pixel_to_board, polar_to_field


@pytest.fixture
def board() -> BoardModel:
    """Standard board model with identity-like homography."""
    # Identity homography mapping pixels directly (for testing pixel_to_board
    # we use a separate test with known homography)
    return BoardModel(
        homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )


class TestBoardToPolar:
    def test_center_is_zero(self) -> None:
        r, a = board_to_polar((0.0, 0.0))
        assert r == pytest.approx(0.0)

    def test_straight_up(self) -> None:
        """Point directly above center (negative y = 12 o'clock)."""
        r, a = board_to_polar((0.0, -100.0))
        assert r == pytest.approx(100.0)
        assert a == pytest.approx(0.0, abs=0.1)

    def test_straight_right(self) -> None:
        """Point to the right = 3 o'clock = 90 degrees."""
        r, a = board_to_polar((100.0, 0.0))
        assert r == pytest.approx(100.0)
        assert a == pytest.approx(90.0, abs=0.1)

    def test_straight_down(self) -> None:
        """Point below center = 6 o'clock = 180 degrees."""
        r, a = board_to_polar((0.0, 100.0))
        assert r == pytest.approx(100.0)
        assert a == pytest.approx(180.0, abs=0.1)

    def test_straight_left(self) -> None:
        """Point to the left = 9 o'clock = 270 degrees."""
        r, a = board_to_polar((-100.0, 0.0))
        assert r == pytest.approx(100.0)
        assert a == pytest.approx(270.0, abs=0.1)


class TestPolarToField:
    def test_bullseye(self, board: BoardModel) -> None:
        assert polar_to_field(5.0, 0.0, board) == "BULL"

    def test_outer_bull(self, board: BoardModel) -> None:
        assert polar_to_field(10.0, 0.0, board) == "25"

    def test_miss(self, board: BoardModel) -> None:
        assert polar_to_field(200.0, 45.0, board) == "MISS"

    def test_triple_20(self, board: BoardModel) -> None:
        """Triple 20 is at 12 o'clock (0 degrees), radius ~103mm."""
        field = polar_to_field(103.0, 0.0, board)
        assert field == "T20"

    def test_double_20(self, board: BoardModel) -> None:
        """Double 20 is at 12 o'clock, radius ~166mm."""
        field = polar_to_field(166.0, 0.0, board)
        assert field == "D20"

    def test_single_inner_20(self, board: BoardModel) -> None:
        """Inner single 20 at 12 o'clock, radius ~50mm."""
        field = polar_to_field(50.0, 0.0, board)
        assert field == "S20"

    def test_single_outer_20(self, board: BoardModel) -> None:
        """Outer single 20 at 12 o'clock, radius ~130mm."""
        field = polar_to_field(130.0, 0.0, board)
        assert field == "S20"

    def test_sector_1_at_18_deg(self, board: BoardModel) -> None:
        """Sector 1 is the next sector clockwise from 20 (at 18 degrees)."""
        field = polar_to_field(103.0, 18.0, board)
        assert field == "T1"

    def test_sector_5_at_342_deg(self, board: BoardModel) -> None:
        """Sector 5 is the last sector counter-clockwise from 20 (at 342 degrees)."""
        field = polar_to_field(103.0, 342.0, board)
        assert field == "T5"

    def test_double_16(self, board: BoardModel) -> None:
        """Double 16 is at approx 252 degrees (sector index 15)."""
        # Sector 16 is at index 13 in SECTOR_ORDER -> 13 * 18 = 234 degrees center
        field = polar_to_field(166.0, 234.0, board)
        assert field == "D16"


class TestGetScore:
    def test_bull(self) -> None:
        assert get_score("BULL") == (50, 1)

    def test_outer_bull(self) -> None:
        assert get_score("25") == (25, 1)

    def test_miss(self) -> None:
        assert get_score("MISS") == (0, 0)

    def test_triple_20(self) -> None:
        assert get_score("T20") == (20, 3)

    def test_double_16(self) -> None:
        assert get_score("D16") == (16, 2)

    def test_single_5(self) -> None:
        assert get_score("S5") == (5, 1)

    def test_invalid(self) -> None:
        assert get_score("X99") == (0, 0)

    def test_empty(self) -> None:
        assert get_score("") == (0, 0)


class TestPixelToBoard:
    def test_center_maps_to_origin(self) -> None:
        """With identity homography and output_size=500, pixel (250,250) -> (0,0)."""
        H = np.eye(3, dtype=np.float64)
        x, y = pixel_to_board((250.0, 250.0), H, output_size=500)
        assert x == pytest.approx(0.0, abs=0.1)
        assert y == pytest.approx(0.0, abs=0.1)

    def test_edge_maps_to_board_radius(self) -> None:
        """Pixel (500, 250) -> right edge -> (170, 0)mm."""
        H = np.eye(3, dtype=np.float64)
        x, y = pixel_to_board((500.0, 250.0), H, output_size=500)
        assert x == pytest.approx(170.0, abs=0.1)
        assert y == pytest.approx(0.0, abs=0.1)
