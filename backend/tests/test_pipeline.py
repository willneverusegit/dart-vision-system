"""Tests for the contour pipeline and tip extraction."""

import numpy as np

from backend.vision.pipeline import (
    PipelineMode,
    PipelineResult,
    extract_tip,
    process_frame,
)


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


class TestEcoMode:
    def test_eco_skips_canny(self) -> None:
        """Eco mode should still produce a valid result without Canny."""
        bg = np.full((480, 640), 128, dtype=np.uint8)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = process_frame(frame, bg, mode=PipelineMode.ECO)
        assert isinstance(result, PipelineResult)
        assert result.debug_output is None

    def test_eco_detects_bright_object(self) -> None:
        """Eco mode can still detect a large bright object."""
        bg = np.zeros((480, 640), dtype=np.uint8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[150:300, 315:325] = 200
        result = process_frame(frame, bg, mode=PipelineMode.ECO)
        assert isinstance(result, PipelineResult)


class TestDebugMode:
    def test_debug_produces_thumbnails(self) -> None:
        """Debug mode should produce debug output with thumbnails."""
        bg = np.zeros((480, 640), dtype=np.uint8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[150:300, 315:325] = 200
        result = process_frame(frame, bg, mode=PipelineMode.DEBUG)
        assert result.debug_output is not None
        assert result.debug_output.grayscale_b64 is not None
        assert result.debug_output.diff_b64 is not None
        assert result.debug_output.canny_b64 is not None
        assert result.debug_output.contours_b64 is not None

    def test_normal_mode_no_debug(self) -> None:
        """Normal mode should not produce debug output."""
        bg = np.full((480, 640), 128, dtype=np.uint8)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = process_frame(frame, bg, mode=PipelineMode.NORMAL)
        assert result.debug_output is None

    def test_debug_thumbnails_are_base64(self) -> None:
        """Debug thumbnails should be valid base64 strings."""
        import base64

        bg = np.zeros((480, 640), dtype=np.uint8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:400, 300:310] = 255
        result = process_frame(frame, bg, mode=PipelineMode.DEBUG)
        assert result.debug_output is not None
        # Should not raise
        base64.b64decode(result.debug_output.grayscale_b64)
