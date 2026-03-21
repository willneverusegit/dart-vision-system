"""Tests for MultiCameraManager."""

import logging
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.vision.multicam import MultiCameraManager

FAKE_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def manager() -> MultiCameraManager:
    """Create a fresh MultiCameraManager for each test."""
    return MultiCameraManager()


@patch("backend.vision.multicam.camera_manager")
class TestAddRemoveCamera:
    """Tests for adding and removing cameras."""

    def test_add_camera_success(self, mock_cm: MagicMock, manager: MultiCameraManager) -> None:
        mock_cm.open.return_value = True
        assert manager.add_camera(0) is True
        assert 0 in manager.get_camera_ids()
        mock_cm.open.assert_called_once_with(0, 640, 480)

    def test_add_camera_failure(self, mock_cm: MagicMock, manager: MultiCameraManager) -> None:
        mock_cm.open.return_value = False
        assert manager.add_camera(1) is False
        assert 1 not in manager.get_camera_ids()

    def test_add_camera_duplicate(self, mock_cm: MagicMock, manager: MultiCameraManager) -> None:
        mock_cm.open.return_value = True
        manager.add_camera(0)
        result = manager.add_camera(0)
        assert result is True
        mock_cm.open.assert_called_once()

    def test_add_camera_custom_resolution(
        self, mock_cm: MagicMock, manager: MultiCameraManager
    ) -> None:
        mock_cm.open.return_value = True
        manager.add_camera(0, width=1280, height=720)
        mock_cm.open.assert_called_once_with(0, 1280, 720)

    def test_remove_camera(self, mock_cm: MagicMock, manager: MultiCameraManager) -> None:
        mock_cm.open.return_value = True
        manager.add_camera(0)
        manager.remove_camera(0)
        assert 0 not in manager.get_camera_ids()
        mock_cm.close.assert_called_once_with(0)

    def test_remove_nonexistent_camera(
        self, mock_cm: MagicMock, manager: MultiCameraManager
    ) -> None:
        manager.remove_camera(99)
        mock_cm.close.assert_not_called()


@patch("backend.vision.multicam.camera_manager")
class TestCaptureSynchronized:
    """Tests for synchronized frame capture."""

    def test_capture_returns_frames_for_all_cameras(
        self, mock_cm: MagicMock, manager: MultiCameraManager
    ) -> None:
        mock_cm.open.return_value = True
        mock_cm.capture_frame.return_value = FAKE_FRAME.copy()
        manager.add_camera(0)
        manager.add_camera(1)

        result = manager.capture_synchronized()

        assert set(result.keys()) == {0, 1}
        for cam_id in (0, 1):
            frame, ts = result[cam_id]
            assert frame.shape == (480, 640, 3)
            assert isinstance(ts, float)

    def test_capture_omits_failed_cameras(
        self, mock_cm: MagicMock, manager: MultiCameraManager
    ) -> None:
        mock_cm.open.return_value = True
        mock_cm.capture_frame.side_effect = lambda idx: FAKE_FRAME.copy() if idx == 0 else None
        manager.add_camera(0)
        manager.add_camera(1)

        result = manager.capture_synchronized()

        assert 0 in result
        assert 1 not in result

    def test_capture_logs_warning_when_time_diff_exceeded(
        self, mock_cm: MagicMock, manager: MultiCameraManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_cm.open.return_value = True

        call_count = 0

        def slow_capture(idx: int) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                time.sleep(0.1)  # 100ms delay
            return FAKE_FRAME.copy()

        mock_cm.capture_frame.side_effect = slow_capture
        manager.add_camera(0)
        manager.add_camera(1)

        with caplog.at_level(logging.WARNING, logger="backend.vision.multicam"):
            manager.capture_synchronized(max_time_diff_ms=10.0)

        assert any("sync exceeded" in r.message.lower() for r in caplog.records)

    def test_capture_no_warning_within_threshold(
        self, mock_cm: MagicMock, manager: MultiCameraManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_cm.open.return_value = True
        mock_cm.capture_frame.return_value = FAKE_FRAME.copy()
        manager.add_camera(0)
        manager.add_camera(1)

        with caplog.at_level(logging.WARNING, logger="backend.vision.multicam"):
            manager.capture_synchronized(max_time_diff_ms=5000.0)

        assert not any("sync exceeded" in r.message.lower() for r in caplog.records)


@patch("backend.vision.multicam.camera_manager")
class TestHealthCheck:
    """Tests for camera health checking."""

    def test_is_healthy_true(self, mock_cm: MagicMock, manager: MultiCameraManager) -> None:
        mock_cm.capture_frame.return_value = FAKE_FRAME.copy()
        assert manager.is_healthy(0) is True

    def test_is_healthy_false(self, mock_cm: MagicMock, manager: MultiCameraManager) -> None:
        mock_cm.capture_frame.return_value = None
        assert manager.is_healthy(0) is False

    def test_health_check_all_cameras(
        self, mock_cm: MagicMock, manager: MultiCameraManager
    ) -> None:
        mock_cm.open.return_value = True
        mock_cm.capture_frame.side_effect = lambda idx: FAKE_FRAME.copy() if idx == 0 else None
        manager.add_camera(0)
        manager.add_camera(1)

        status = manager.health_check()

        assert status == {0: True, 1: False}


@patch("backend.vision.multicam.camera_manager")
class TestCloseAll:
    """Tests for closing all cameras."""

    def test_close_all(self, mock_cm: MagicMock, manager: MultiCameraManager) -> None:
        mock_cm.open.return_value = True
        manager.add_camera(0)
        manager.add_camera(1)
        manager.add_camera(2)

        manager.close_all()

        assert manager.get_camera_ids() == []
        assert mock_cm.close.call_count == 3
