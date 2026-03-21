"""Tests for confidence scoring module."""

import pytest

from backend.scoring.confidence import (
    REVIEW_THRESHOLD,
    calculate_confidence,
    detection_quality_confidence,
    needs_review,
    triangulation_confidence,
    wire_distance_confidence,
)


class TestWireDistanceConfidence:
    def test_center_bull_high_confidence(self):
        """Dead center of bull — far from any wire."""
        conf = wire_distance_confidence(0.0, 0.0)
        assert conf > 0.8

    def test_on_bull_wire(self):
        """Exactly on the bull ring wire boundary."""
        conf = wire_distance_confidence(6.35, 0.0)
        assert conf == 0.0

    def test_near_bull_wire(self):
        """Just inside the wire zone."""
        conf = wire_distance_confidence(6.0, 0.0)
        assert conf < 0.5

    def test_far_from_wire_single(self):
        """Middle of inner single area — far from wires."""
        conf = wire_distance_confidence(55.0, 0.0)  # centered in sector 20
        assert conf > 0.8

    def test_on_triple_wire(self):
        """On the triple ring boundary."""
        conf = wire_distance_confidence(99.0, 0.0)
        assert conf == 0.0

    def test_on_double_wire(self):
        """On the double ring boundary."""
        conf = wire_distance_confidence(170.0, 0.0)
        assert conf == 0.0

    def test_near_sector_wire(self):
        """Near a sector boundary line."""
        # Sector boundaries at 9 degrees — test at 9.5 degrees (very close)
        conf = wire_distance_confidence(130.0, 9.2)
        assert conf < 0.5

    def test_sector_wire_irrelevant_in_bull(self):
        """Sector wires don't exist in the bull area."""
        # At 9 degrees (sector boundary) but in bull area
        conf = wire_distance_confidence(10.0, 9.0)
        # Should still be reasonable — only ring wire matters
        assert conf > 0.3

    def test_outside_board(self):
        """Outside the board — sector wires irrelevant."""
        conf = wire_distance_confidence(200.0, 9.0)
        # Far from double_outer ring at 170mm
        assert conf > 0.5


class TestDetectionQualityConfidence:
    def test_no_metrics_default(self):
        """Default confidence when no detection metrics available."""
        assert detection_quality_confidence() == 0.8

    def test_good_contour(self):
        """Good contour area gives high confidence."""
        conf = detection_quality_confidence(contour_area=1000)
        assert conf == 1.0

    def test_too_small_contour(self):
        """Very small contour (noise) gives low confidence."""
        conf = detection_quality_confidence(contour_area=50)
        assert conf < 0.5

    def test_too_large_contour(self):
        """Very large contour gives low confidence."""
        conf = detection_quality_confidence(contour_area=10000)
        assert conf < 0.5

    def test_good_solidity(self):
        """High solidity gives high confidence."""
        conf = detection_quality_confidence(contour_solidity=0.9)
        assert conf > 0.8

    def test_low_solidity(self):
        """Low solidity gives low confidence."""
        conf = detection_quality_confidence(contour_solidity=0.2)
        assert conf < 0.5

    def test_combined_metrics(self):
        """Both metrics combined — averaged."""
        conf = detection_quality_confidence(contour_area=1000, contour_solidity=0.8)
        assert conf > 0.8


class TestTriangulationConfidence:
    def test_no_stereo_default(self):
        """Default for single-camera mode."""
        assert triangulation_confidence() == 0.8

    def test_zero_error(self):
        """Perfect triangulation."""
        assert triangulation_confidence(0.0) == 1.0

    def test_small_error(self):
        """Small reprojection error."""
        conf = triangulation_confidence(1.0)
        assert conf == pytest.approx(0.8)

    def test_large_error(self):
        """Large error gives zero confidence."""
        assert triangulation_confidence(5.0) == 0.0
        assert triangulation_confidence(10.0) == 0.0


class TestOverallConfidence:
    def test_all_perfect(self):
        """All factors at maximum."""
        conf = calculate_confidence(
            radius_mm=55.0,  # middle of inner single
            angle_deg=0.0,  # center of sector 20
            contour_area=1000,
            contour_solidity=0.9,
            reprojection_error=0.0,
        )
        assert conf > 0.9

    def test_all_bad(self):
        """All factors at minimum."""
        conf = calculate_confidence(
            radius_mm=99.0,  # on triple wire
            angle_deg=9.0,  # on sector wire
            contour_area=50,  # too small
            contour_solidity=0.1,
            reprojection_error=10.0,
        )
        assert conf < 0.2

    def test_defaults_reasonable(self):
        """With only position data, confidence should be reasonable."""
        conf = calculate_confidence(radius_mm=55.0, angle_deg=0.0)
        assert 0.5 < conf < 1.0

    def test_bounded_0_1(self):
        """Confidence is always between 0 and 1."""
        for r in [0, 6.35, 50, 99, 107, 130, 162, 170, 200]:
            for a in [0, 9, 18, 45, 90, 180, 270, 359]:
                conf = calculate_confidence(radius_mm=r, angle_deg=a)
                assert 0.0 <= conf <= 1.0


class TestNeedsReview:
    def test_high_confidence_no_review(self):
        assert needs_review(0.9) is False

    def test_low_confidence_needs_review(self):
        assert needs_review(0.3) is True

    def test_threshold_boundary(self):
        assert needs_review(REVIEW_THRESHOLD) is False
        assert needs_review(REVIEW_THRESHOLD - 0.01) is True
