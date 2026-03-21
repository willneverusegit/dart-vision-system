"""Tests for the warp module and API endpoints."""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.vision.warp import WarpEngine

client = TestClient(app)

# A simple identity-like homography (no perspective distortion)
DUMMY_H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _synthetic_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic BGR frame for testing."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


# --- WarpEngine unit tests ---


class TestWarpEngine:
    """Tests for the WarpEngine class."""

    def test_not_configured_initially(self) -> None:
        engine = WarpEngine()
        assert engine.is_configured() is False

    def test_configure_and_check(self) -> None:
        engine = WarpEngine()
        engine.set_homography(np.eye(3), output_size=400)
        assert engine.is_configured() is True
        assert engine._output_size == 400

    def test_invalid_homography_shape(self) -> None:
        engine = WarpEngine()
        with pytest.raises(ValueError, match="3x3"):
            engine.set_homography(np.eye(2))

    def test_warp_frame_without_config(self) -> None:
        engine = WarpEngine()
        with pytest.raises(RuntimeError, match="not configured"):
            engine.warp_frame(_synthetic_frame())

    def test_warp_frame(self) -> None:
        engine = WarpEngine()
        engine.set_homography(np.eye(3), output_size=200)
        result = engine.warp_frame(_synthetic_frame())
        assert result.shape == (200, 200, 3)

    def test_warp_frame_jpeg(self) -> None:
        engine = WarpEngine()
        engine.set_homography(np.eye(3), output_size=200)
        jpeg = engine.warp_frame_jpeg(_synthetic_frame())
        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
        # JPEG magic bytes
        assert jpeg[:2] == b"\xff\xd8"


# --- API tests ---


class TestWarpAPI:
    """Tests for the /api/warp endpoints."""

    def test_status_unconfigured(self) -> None:
        # Reset the singleton for a clean state
        from backend.vision.warp import warp_engine

        warp_engine._homography = None
        resp = client.get("/api/warp/status")
        assert resp.status_code == 200
        assert resp.json()["configured"] is False

    def test_configure_and_status(self) -> None:
        from backend.vision.warp import warp_engine

        warp_engine._homography = None

        resp = client.post(
            "/api/warp/configure",
            json={"homography": DUMMY_H, "output_size": 300},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is True
        assert data["output_size"] == 300

        resp = client.get("/api/warp/status")
        assert resp.json()["configured"] is True

    def test_configure_invalid_matrix(self) -> None:
        resp = client.post(
            "/api/warp/configure",
            json={"homography": [[1, 0], [0, 1]]},
        )
        assert resp.status_code == 422

    def test_frame_without_config(self) -> None:
        from backend.vision.warp import warp_engine

        warp_engine._homography = None

        resp = client.get("/api/warp/frame/0")
        assert resp.status_code == 400

    def test_frame_with_mock_camera(self) -> None:
        from backend.vision.warp import warp_engine

        warp_engine.set_homography(np.eye(3), output_size=200)

        fake_frame = _synthetic_frame()
        with patch(
            "backend.api.warp.camera_manager.capture_frame",
            return_value=fake_frame,
        ):
            resp = client.get("/api/warp/frame/0")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content[:2] == b"\xff\xd8"

    def test_frame_camera_unavailable(self) -> None:
        from backend.vision.warp import warp_engine

        warp_engine.set_homography(np.eye(3))

        with patch(
            "backend.api.warp.camera_manager.capture_frame",
            return_value=None,
        ):
            resp = client.get("/api/warp/frame/0")
        assert resp.status_code == 404
