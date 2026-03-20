import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(autouse=True)
def reset_game_state():
    import backend.api.game as game_module

    game_module._current_game = None
    yield
    game_module._current_game = None


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_start_game(client):
    resp = client.post("/api/game/start", json={"player_names": ["Alice", "Bob"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] is True
    assert len(data["players"]) == 2
    assert data["players"][0]["name"] == "Alice"


def test_start_game_conflict(client):
    client.post("/api/game/start", json={"player_names": ["Alice"]})
    resp = client.post("/api/game/start", json={"player_names": ["Bob"]})
    assert resp.status_code == 409


def test_get_state(client):
    client.post("/api/game/start", json={"player_names": ["Alice"]})
    resp = client.get("/api/game/state")
    assert resp.status_code == 200
    assert resp.json()["active"] is True


def test_get_state_no_game(client):
    resp = client.get("/api/game/state")
    assert resp.status_code == 404


def test_throw_and_stop(client):
    client.post("/api/game/start", json={"player_names": ["Alice"], "mode": "free"})
    resp = client.post("/api/game/throw", json={"score": 60, "field": "T20"})
    assert resp.status_code == 200
    assert resp.json()["history"][0]["score"] == 60

    resp = client.post("/api/game/stop")
    assert resp.status_code == 200
    assert len(resp.json()["history"]) == 1


def test_websocket_echo(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text("hello")
        data = ws.receive_json()
        assert data["type"] == "echo"
        assert data["data"] == "hello"
