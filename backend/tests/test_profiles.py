"""Tests for profile CRUD API and profile_store service."""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import profile_store

TEST_PROFILES_DIR = Path(__file__).parent / "_test_profiles"


@pytest.fixture(autouse=True)
def isolated_profiles_dir(monkeypatch):
    """Redirect profile storage to a temp directory for each test."""
    TEST_PROFILES_DIR.mkdir(exist_ok=True)
    monkeypatch.setattr(profile_store, "PROFILES_DIR", TEST_PROFILES_DIR)
    monkeypatch.setattr(profile_store, "ACTIVE_FILE", TEST_PROFILES_DIR / ".active")
    yield
    shutil.rmtree(TEST_PROFILES_DIR, ignore_errors=True)


@pytest.fixture
def client():
    return TestClient(app)


SAMPLE_PROFILE = {
    "id": "cam1",
    "role": "left",
    "resolution": [1920, 1080],
}


# --- API tests ---


def test_list_empty(client):
    resp = client.get("/api/profiles")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_and_get(client):
    resp = client.post("/api/profiles", json=SAMPLE_PROFILE)
    assert resp.status_code == 201
    assert resp.json()["id"] == "cam1"

    resp = client.get("/api/profiles/cam1")
    assert resp.status_code == 200
    assert resp.json()["role"] == "left"


def test_list_shows_active(client):
    client.post("/api/profiles", json=SAMPLE_PROFILE)
    profiles = client.get("/api/profiles").json()
    assert len(profiles) == 1
    assert profiles[0]["active"] is True


def test_get_not_found(client):
    resp = client.get("/api/profiles/nope")
    assert resp.status_code == 404


def test_delete(client):
    client.post("/api/profiles", json=SAMPLE_PROFILE)
    resp = client.delete("/api/profiles/cam1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "cam1"

    resp = client.get("/api/profiles")
    assert resp.json() == []


def test_delete_not_found(client):
    resp = client.delete("/api/profiles/nope")
    assert resp.status_code == 404


# --- Service tests ---


def test_save_and_load():
    from backend.models.camera import CameraProfile

    p = CameraProfile(id="test1", role="top", resolution=(640, 480))
    profile_store.save_profile(p)
    loaded = profile_store.load_profile("test1")
    assert loaded is not None
    assert loaded.id == "test1"
    assert loaded.role == "top"


def test_active_profile():
    profile_store.set_active("cam1")
    assert profile_store.get_active_profile_id() == "cam1"


def test_no_active_profile():
    assert profile_store.get_active_profile_id() is None


def test_get_last_profile_none():
    assert profile_store.get_last_profile() is None


def test_delete_clears_active():
    from backend.models.camera import CameraProfile

    p = CameraProfile(id="cam1", role="left", resolution=(1920, 1080))
    profile_store.save_profile(p)
    profile_store.set_active("cam1")
    profile_store.delete_profile("cam1")
    assert profile_store.get_active_profile_id() is None
