"""REST endpoints for camera profile CRUD."""

from fastapi import APIRouter, HTTPException

from backend.models.camera import CameraProfile
from backend.services.profile_store import (
    delete_profile,
    get_active_profile_id,
    list_profiles,
    load_profile,
    save_profile,
    set_active,
)

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


@router.get("")
def get_profiles() -> list[dict]:
    """List all saved profiles with their IDs and active status."""
    active_id = get_active_profile_id()
    return [{"id": pid, "active": pid == active_id} for pid in list_profiles()]


@router.get("/{profile_id}")
def get_profile(profile_id: str) -> CameraProfile:
    """Load a single profile by ID."""
    profile = load_profile(profile_id)
    if profile is None:
        raise HTTPException(404, f"Profile '{profile_id}' not found")
    return profile


@router.post("", status_code=201)
def create_profile(profile: CameraProfile) -> CameraProfile:
    """Save a new or updated profile."""
    save_profile(profile)
    set_active(profile.id)
    return profile


@router.delete("/{profile_id}")
def remove_profile(profile_id: str) -> dict:
    """Delete a profile by ID."""
    if not delete_profile(profile_id):
        raise HTTPException(404, f"Profile '{profile_id}' not found")
    return {"deleted": profile_id}
