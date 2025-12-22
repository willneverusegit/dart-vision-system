"""
Unit tests for board module.
"""
import numpy as np
import pytest
from src.core import BoardGeometry, Hit
from src.board import DartboardMapper, BoardVisualizer


def test_dartboard_mapper_init():
    """Test DartboardMapper initialization."""
    mapper = DartboardMapper(
        board_center=(400.0, 400.0),
        mm_per_pixel=0.5
    )

    assert mapper.center == (400.0, 400.0)
    assert mapper.mm_per_pixel == 0.5


def test_pixel_to_polar():
    """Test pixel to polar conversion."""
    mapper = DartboardMapper(
        board_center=(400.0, 400.0),
        mm_per_pixel=0.5
    )

    # Test top (0°)
    radius, angle = mapper.pixel_to_polar(400, 300)
    assert radius == pytest.approx(100.0, abs=0.1)
    assert angle == pytest.approx(0.0, abs=1.0)

    # Test right (90°)
    radius, angle = mapper.pixel_to_polar(500, 400)
    assert radius == pytest.approx(100.0, abs=0.1)
    assert angle == pytest.approx(90.0, abs=1.0)

    # Test bottom (180°)
    radius, angle = mapper.pixel_to_polar(400, 500)
    assert radius == pytest.approx(100.0, abs=0.1)
    assert angle == pytest.approx(180.0, abs=1.0)

    # Test left (270°)
    radius, angle = mapper.pixel_to_polar(300, 400)
    assert radius == pytest.approx(100.0, abs=0.1)
    assert angle == pytest.approx(270.0, abs=1.0)


def test_angle_to_sector():
    """Test angle to sector mapping."""
    mapper = DartboardMapper()

    # Sector layout (each 18° wide):
    # Sector 20: centered at 0°, range [-9°, 9°)
    # Sector 1:  centered at 18°, range [9°, 27°)
    # Sector 18: centered at 36°, range [27°, 45°)
    # Sector 4:  centered at 54°, range [45°, 63°)

    # Test sector 20 (top, centered at 0°)
    assert mapper.angle_to_sector(0) == 20
    assert mapper.angle_to_sector(5) == 20
    assert mapper.angle_to_sector(8) == 20

    # Test sector 1 (after 20, clockwise)
    assert mapper.angle_to_sector(9) == 1
    assert mapper.angle_to_sector(18) == 1
    assert mapper.angle_to_sector(26) == 1

    # Test sector 18 (after 1)
    assert mapper.angle_to_sector(27) == 18
    assert mapper.angle_to_sector(36) == 18

    # Test sector 4 (after 18)
    assert mapper.angle_to_sector(45) == 4
    assert mapper.angle_to_sector(54) == 4

    # Test wrapping (sector 5 is last, then wraps to 20)
    assert mapper.angle_to_sector(351) == 20  # Just before 0°
    assert mapper.angle_to_sector(359) == 20


def test_radius_to_ring():
    """Test radius to ring mapping."""
    # Setup with known scale
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5  # 0.5 mm/px
    )

    # Double bull: 6.35 mm radius = 12.7 px
    ring, mult = mapper.radius_to_ring(10.0)
    assert ring == "double_bull"
    assert mult == 50

    # Single bull: 15.9 mm radius = 31.8 px
    ring, mult = mapper.radius_to_ring(25.0)
    assert ring == "single_bull"
    assert mult == 25

    # Triple ring: ~99-107 mm = 198-214 px
    ring, mult = mapper.radius_to_ring(200.0)
    assert ring == "triple"
    assert mult == 3

    # Double ring: ~162-170 mm = 324-340 px
    ring, mult = mapper.radius_to_ring(330.0)
    assert ring == "double"
    assert mult == 2

    # Single (inner)
    ring, mult = mapper.radius_to_ring(100.0)
    assert ring == "single"
    assert mult == 1

    # Miss (outside board)
    ring, mult = mapper.radius_to_ring(500.0)
    assert ring == "miss"
    assert mult == 0


def test_pixel_to_score():
    """Test full pixel to score conversion."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )

    # Center (double bull)
    hit = mapper.pixel_to_score(400, 400)
    assert hit.score == 50
    assert hit.multiplier == 50

    # Triple 20 (top, ~200px out)
    hit = mapper.pixel_to_score(400, 400 - 200)
    assert hit.sector == 20
    assert hit.multiplier == 3
    assert hit.score == 60


def test_is_valid_hit():
    """Test hit validation."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )

    # Inside board
    assert mapper.is_valid_hit(400, 400)
    assert mapper.is_valid_hit(400, 300)

    # Outside board
    assert not mapper.is_valid_hit(100, 100)


def test_get_ring_boundaries():
    """Test ring boundaries retrieval."""
    mapper = DartboardMapper(mm_per_pixel=0.5)
    boundaries = mapper.get_ring_boundaries()

    assert "inner_bull" in boundaries
    assert "double_outer" in boundaries
    assert boundaries["inner_bull"] < boundaries["outer_bull"]


def test_get_sector_boundaries():
    """Test sector boundaries retrieval."""
    mapper = DartboardMapper()
    boundaries = mapper.get_sector_boundaries()

    assert len(boundaries) == 20
    assert boundaries[0][0] == 20  # First sector is 20


def test_board_visualizer_init():
    """Test BoardVisualizer initialization."""
    mapper = DartboardMapper()
    viz = BoardVisualizer(mapper, opacity=0.5)

    assert viz.mapper == mapper
    assert viz.opacity == 0.5


def test_draw_board_overlay():
    """Test board overlay drawing."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )
    viz = BoardVisualizer(mapper)

    # Create test image
    image = np.zeros((800, 800, 3), dtype=np.uint8)

    # Draw overlay (should not crash)
    result = viz.draw_board_overlay(image)

    assert result.shape == image.shape
    assert not np.array_equal(result, image)  # Image should be modified


def test_draw_hit():
    """Test hit marker drawing."""
    mapper = DartboardMapper()
    viz = BoardVisualizer(mapper)

    # Create test image and hit
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    hit = Hit(
        x_px=400, y_px=400,
        radius=0, angle=0,
        sector=20, multiplier=3, score=60
    )

    # Draw hit (should not crash)
    result = viz.draw_hit(image, hit)

    assert result.shape == image.shape
    assert not np.array_equal(result, image)


if __name__ == "__main__":
    print("Running board module tests...")
    test_dartboard_mapper_init()
    print("✓ DartboardMapper init test passed")
    test_pixel_to_polar()
    print("✓ Pixel to polar test passed")
    test_angle_to_sector()
    print("✓ Angle to sector test passed")
    test_radius_to_ring()
    print("✓ Radius to ring test passed")
    test_pixel_to_score()
    print("✓ Pixel to score test passed")
    test_is_valid_hit()
    print("✓ Hit validation test passed")
    test_get_ring_boundaries()
    print("✓ Ring boundaries test passed")
    test_get_sector_boundaries()
    print("✓ Sector boundaries test passed")
    test_board_visualizer_init()
    print("✓ BoardVisualizer init test passed")
    test_draw_board_overlay()
    print("✓ Board overlay test passed")
    test_draw_hit()
    print("✓ Hit drawing test passed")
    print("\n✓ All board tests passed!")