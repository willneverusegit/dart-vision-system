"""Tests for background subtraction and motion detection."""

import numpy as np

from backend.vision.detection import BackgroundModel, detect_motion


class TestBackgroundModel:
    def test_initial_background_is_none(self) -> None:
        bg = BackgroundModel()
        assert bg.get_background() is None

    def test_set_background_bgr(self) -> None:
        bg = BackgroundModel()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bg.set_background(frame)
        result = bg.get_background()
        assert result is not None
        assert len(result.shape) == 2  # grayscale

    def test_set_background_gray(self) -> None:
        bg = BackgroundModel()
        frame = np.zeros((480, 640), dtype=np.uint8)
        bg.set_background(frame)
        result = bg.get_background()
        assert result is not None
        assert result.shape == (480, 640)


class TestDetectMotion:
    def test_no_motion_identical_frames(self) -> None:
        bg = np.full((480, 640), 128, dtype=np.uint8)
        frame = np.full((480, 640), 128, dtype=np.uint8)
        assert detect_motion(frame, bg) is False

    def test_motion_with_rectangle(self) -> None:
        """A white rectangle on black background should be detected as motion."""
        bg = np.zeros((480, 640), dtype=np.uint8)
        frame = np.zeros((480, 640), dtype=np.uint8)
        # Draw a large white rectangle (simulating a dart)
        frame[200:280, 300:320] = 255
        assert detect_motion(frame, bg, threshold=25, min_area=500) is True

    def test_no_motion_below_threshold(self) -> None:
        """Small intensity changes below threshold should not trigger."""
        bg = np.full((480, 640), 128, dtype=np.uint8)
        frame = np.full((480, 640), 140, dtype=np.uint8)  # diff=12 < threshold=25
        assert detect_motion(frame, bg, threshold=25) is False

    def test_motion_with_bgr_frame(self) -> None:
        """Should handle BGR frames by converting to grayscale."""
        bg = np.zeros((480, 640), dtype=np.uint8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[200:280, 300:320] = 255  # White rectangle in BGR
        assert detect_motion(frame, bg, threshold=25, min_area=500) is True

    def test_small_area_not_detected(self) -> None:
        """Motion below min_area should not be detected."""
        bg = np.zeros((480, 640), dtype=np.uint8)
        frame = np.zeros((480, 640), dtype=np.uint8)
        # Tiny 5x5 white square (area=25 < min_area=500)
        frame[200:205, 300:305] = 255
        assert detect_motion(frame, bg, threshold=25, min_area=500) is False
