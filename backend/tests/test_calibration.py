"""Tests for calibration module and API."""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.camera import CameraProfile, CameraRole
from backend.vision.calibration import (
    CalibrationSession,
    list_profiles,
    load_profile,
    save_profile,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_sessions():
    import backend.api.calibration as cal_api

    cal_api._sessions.clear()
    yield
    cal_api._sessions.clear()


class TestCalibrationSession:
    def test_initial_state(self):
        session = CalibrationSession("cam0", CameraRole.LEFT)
        assert session.frame_count == 0
        assert session.camera_matrix is None

    def test_process_frame_no_markers(self):
        session = CalibrationSession("cam0")
        # Black image — no markers
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = session.process_frame(frame)
        assert result["markers_detected"] == 0
        assert result["frames_collected"] == 0
        assert result["ready"] is False

    def test_calibrate_not_enough_frames(self):
        session = CalibrationSession("cam0")
        result = session.calibrate()
        assert result is None

    def test_get_profile_without_calibration(self):
        session = CalibrationSession("cam0", CameraRole.RIGHT)
        profile = session.get_profile(resolution=(640, 480))
        assert profile.id == "cam0"
        assert profile.role == CameraRole.RIGHT
        assert profile.intrinsics is None


class TestProfilePersistence:
    def test_save_and_load(self, tmp_path):
        profile = CameraProfile(
            id="cam0",
            role=CameraRole.LEFT,
            resolution=(640, 480),
        )
        with patch("backend.vision.calibration.PROFILES_DIR", tmp_path):
            path = save_profile(profile, "test_profile")
            assert path.exists()

            loaded = load_profile("test_profile")
            assert loaded is not None
            assert loaded.id == "cam0"

    def test_load_nonexistent(self, tmp_path):
        with patch("backend.vision.calibration.PROFILES_DIR", tmp_path):
            assert load_profile("nonexistent") is None

    def test_list_profiles(self, tmp_path):
        profile = CameraProfile(id="cam0", role=CameraRole.LEFT, resolution=(640, 480))
        with patch("backend.vision.calibration.PROFILES_DIR", tmp_path):
            save_profile(profile, "profile_a")
            save_profile(profile, "profile_b")
            names = list_profiles()
            assert set(names) == {"profile_a", "profile_b"}


class TestCalibrationAPI:
    def test_start_calibration(self, client):
        resp = client.post(
            "/api/calibrate/start",
            json={"camera_id": "cam0", "role": "left"},
        )
        assert resp.status_code == 200
        assert resp.json()["camera_id"] == "cam0"
        assert resp.json()["frames_collected"] == 0

    def test_status_no_session(self, client):
        resp = client.get("/api/calibrate/status/cam99")
        assert resp.status_code == 404

    def test_status_after_start(self, client):
        client.post("/api/calibrate/start", json={"camera_id": "cam0"})
        resp = client.get("/api/calibrate/status/cam0")
        assert resp.status_code == 200
        assert resp.json()["frames_collected"] == 0

    def test_finish_not_enough_frames(self, client):
        client.post("/api/calibrate/start", json={"camera_id": "cam0"})
        resp = client.post("/api/calibrate/finish", params={"camera_id": "cam0"})
        assert resp.status_code == 400
