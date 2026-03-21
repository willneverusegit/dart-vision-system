"""Tests for the WebSocket ConnectionManager and detection API."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestConnectionManager:
    def test_websocket_connect_and_echo(self, client: TestClient) -> None:
        """WebSocket should accept connections and echo messages."""
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text("hello")
            data = ws.receive_json()
            assert data["type"] == "echo"
            assert data["data"] == "hello"


class TestDetectionAPI:
    def test_status_initially_not_running(self, client: TestClient) -> None:
        resp = client.get("/api/detection/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False

    def test_start_without_background_fails(self, client: TestClient) -> None:
        resp = client.post("/api/detection/start", json={"camera_id": 0})
        assert resp.status_code == 400
        assert "background" in resp.json()["detail"].lower()

    def test_stop_when_not_running(self, client: TestClient) -> None:
        resp = client.post("/api/detection/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
