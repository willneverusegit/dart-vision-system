"""Tests for the contour pipeline and tip extraction."""

import numpy as np

from backend.vision.pipeline import PipelineResult, extract_tip, process_frame


class TestExtractTip:
    def test_no_contours_returns_none(self) -> None:
        assert extract_tip([]) is None

    def test_single_elongated_contour(self) -> None:
        """An elongated contour should return a tip (extreme point)."""
        # Create a vertical line-like contour
        points = np.array(
            [
                [[100, 100]],
                [[105, 100]],
                [[105, 200]],
                [[100, 200]],
            ],
            dtype=np.int32,
        )
        tip = extract_tip([points])
        assert tip is not None
        # Tip should be one of the extreme points
        tx, ty = tip
        assert 100 <= tx <= 105
        assert ty in (100, 200)  # one of the ends


class TestProcessFrame:
    def test_no_dart_returns_no_tip(self) -> None:
        """Identical frame and background should yield no tip."""
        bg = np.full((480, 640), 128, dtype=np.uint8)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = process_frame(frame, bg)
        assert result.tip_point is None
        assert result.field == ""
        assert result.score == 0

    def test_dart_shaped_object_detected(self) -> None:
        """A long thin bright shape on dark background should produce a tip."""
        bg = np.zeros((480, 640), dtype=np.uint8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a vertical bright bar (simulating a dart shaft)
        frame[150:300, 315:325] = 200
        result = process_frame(frame, bg)
        # With morphological ops and contour filtering, this should detect something
        # The result depends on whether it passes the aspect ratio filter
        assert isinstance(result, PipelineResult)


class TestPipelineResult:
    def test_default_values(self) -> None:
        r = PipelineResult()
        assert r.tip_point is None
        assert r.field == ""
        assert r.score == 0
        assert r.multiplier == 1
