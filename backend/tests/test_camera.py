"""Tests for CameraManager (unit tests without real camera hardware)."""

from unittest.mock import MagicMock, patch

import numpy as np

from backend.vision.camera import CameraManager


def test_list_devices_no_cameras():
    """list_devices returns empty when no cameras available."""
    mgr = CameraManager()
    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap.return_value.isOpened.return_value = False
        devices = mgr.list_devices(max_check=2)
    assert devices == []


def test_list_devices_with_camera():
    """list_devices finds available cameras."""
    mgr = CameraManager()
    with patch("cv2.VideoCapture") as mock_cap:
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = True
        mock_cap.return_value = cap_instance
        devices = mgr.list_devices(max_check=2)
    assert len(devices) == 2
    assert devices[0].name == "Camera 0"


def test_open_and_close():
    """Open and close a camera."""
    mgr = CameraManager()
    with patch("cv2.VideoCapture") as mock_cap:
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = True
        mock_cap.return_value = cap_instance

        assert mgr.open(0) is True
        assert mgr.is_open(0) is True

        mgr.close(0)
        assert mgr.is_open(0) is False
        cap_instance.release.assert_called_once()


def test_open_failure():
    """Open returns False when camera can't be opened."""
    mgr = CameraManager()
    with patch("cv2.VideoCapture") as mock_cap:
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = False
        mock_cap.return_value = cap_instance

        assert mgr.open(99) is False
        assert mgr.is_open(99) is False


def test_capture_frame():
    """capture_frame returns numpy array."""
    mgr = CameraManager()
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.VideoCapture") as mock_cap:
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = True
        cap_instance.read.return_value = (True, fake_frame)
        mock_cap.return_value = cap_instance

        mgr.open(0)
        frame = mgr.capture_frame(0)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        mgr.close_all()


def test_capture_frame_not_open():
    """capture_frame returns None for unopened camera."""
    mgr = CameraManager()
    assert mgr.capture_frame(0) is None


def test_capture_frame_jpeg():
    """capture_frame_jpeg returns JPEG bytes."""
    mgr = CameraManager()
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.VideoCapture") as mock_cap:
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = True
        cap_instance.read.return_value = (True, fake_frame)
        mock_cap.return_value = cap_instance

        mgr.open(0)
        jpeg = mgr.capture_frame_jpeg(0)
        assert jpeg is not None
        # JPEG starts with FFD8
        assert jpeg[:2] == b"\xff\xd8"
        mgr.close_all()


def test_close_all():
    """close_all releases all cameras."""
    mgr = CameraManager()
    with patch("cv2.VideoCapture") as mock_cap:
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = True
        mock_cap.return_value = cap_instance

        mgr.open(0)
        mgr.open(1)
        mgr.close_all()
        assert not mgr.is_open(0)
        assert not mgr.is_open(1)


def test_get_resolution():
    """get_resolution returns current camera dimensions."""
    mgr = CameraManager()
    with patch("cv2.VideoCapture") as mock_cap:
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = True
        cap_instance.get.side_effect = lambda prop: {3: 640.0, 4: 480.0}.get(prop, 0.0)
        mock_cap.return_value = cap_instance

        mgr.open(0)
        res = mgr.get_resolution(0)
        assert res == (640, 480)
        mgr.close_all()
