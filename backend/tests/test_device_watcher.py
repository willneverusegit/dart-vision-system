"""Tests for camera DeviceWatcher."""

import time
from unittest.mock import MagicMock

from backend.vision.camera import CameraDevice, CameraManager, DeviceWatcher


class TestDeviceWatcher:
    def test_start_stop(self):
        """Watcher starts and stops cleanly."""
        manager = MagicMock(spec=CameraManager)
        manager.list_devices.return_value = []

        watcher = DeviceWatcher(manager, interval=0.1)
        watcher.start()
        assert watcher.is_running
        watcher.stop()
        assert not watcher.is_running

    def test_initial_known_devices(self):
        """Start populates known indices."""
        manager = MagicMock(spec=CameraManager)
        manager.list_devices.return_value = [
            CameraDevice(index=0, name="Cam 0"),
            CameraDevice(index=1, name="Cam 1"),
        ]

        watcher = DeviceWatcher(manager, interval=0.1)
        watcher.start()
        assert watcher.known_indices == {0, 1}
        watcher.stop()

    def test_device_added_callback(self):
        """Callback fires when a new device appears."""
        manager = MagicMock(spec=CameraManager)
        added_devices = []

        new_dev = [CameraDevice(index=0, name="Cam 0")]
        manager.list_devices.side_effect = [
            [],  # initial
            new_dev,  # first poll
            new_dev,  # subsequent polls
            new_dev,
            new_dev,
        ]

        watcher = DeviceWatcher(
            manager,
            interval=0.1,
            on_added=lambda d: added_devices.append(d),
        )
        watcher.start()
        time.sleep(0.3)  # wait for at least one poll
        watcher.stop()

        assert len(added_devices) >= 1
        assert added_devices[0].index == 0

    def test_device_removed_callback(self):
        """Callback fires when a device disappears."""
        manager = MagicMock(spec=CameraManager)
        removed_ids = []

        manager.list_devices.side_effect = [
            [CameraDevice(index=0, name="Cam 0")],  # initial
            [],  # first poll — device gone
            [],  # subsequent polls
            [],
            [],
        ]

        watcher = DeviceWatcher(
            manager,
            interval=0.1,
            on_removed=lambda idx: removed_ids.append(idx),
        )
        watcher.start()
        time.sleep(0.3)
        watcher.stop()

        assert 0 in removed_ids

    def test_no_change_no_callback(self):
        """No callbacks when device list stays the same."""
        manager = MagicMock(spec=CameraManager)
        added = []
        removed = []

        devices = [CameraDevice(index=0, name="Cam 0")]
        manager.list_devices.return_value = devices

        watcher = DeviceWatcher(
            manager,
            interval=0.1,
            on_added=lambda d: added.append(d),
            on_removed=lambda idx: removed.append(idx),
        )
        watcher.start()
        time.sleep(0.3)
        watcher.stop()

        assert len(added) == 0
        assert len(removed) == 0

    def test_double_start_ignored(self):
        """Calling start twice doesn't create a second thread."""
        manager = MagicMock(spec=CameraManager)
        manager.list_devices.return_value = []

        watcher = DeviceWatcher(manager, interval=0.1)
        watcher.start()
        thread1 = watcher._thread
        watcher.start()
        thread2 = watcher._thread
        assert thread1 is thread2
        watcher.stop()

    def test_callback_exception_does_not_crash(self):
        """Watcher continues even if callback raises."""
        manager = MagicMock(spec=CameraManager)
        manager.list_devices.side_effect = [
            [],
            [CameraDevice(index=0, name="Cam 0")],
            [CameraDevice(index=0, name="Cam 0")],
        ]

        def bad_callback(d):
            raise ValueError("boom")

        watcher = DeviceWatcher(manager, interval=0.1, on_added=bad_callback)
        watcher.start()
        time.sleep(0.3)
        # Should still be running despite callback error
        assert watcher.is_running
        watcher.stop()
