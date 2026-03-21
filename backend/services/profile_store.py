"""Profile persistence — load/save CameraProfile as JSON files."""

from pathlib import Path

from backend.models.camera import CameraProfile

PROFILES_DIR = Path(__file__).parent.parent / "data" / "profiles"
ACTIVE_FILE = PROFILES_DIR / ".active"


def _ensure_dir() -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def list_profiles() -> list[str]:
    """Return sorted list of profile IDs (file stems)."""
    _ensure_dir()
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def load_profile(profile_id: str) -> CameraProfile | None:
    """Load a profile by ID. Returns None if not found."""
    path = PROFILES_DIR / f"{profile_id}.json"
    if not path.exists():
        return None
    return CameraProfile.model_validate_json(path.read_text(encoding="utf-8"))


def save_profile(profile: CameraProfile) -> None:
    """Save a profile to disk (overwrites if exists)."""
    _ensure_dir()
    path = PROFILES_DIR / f"{profile.id}.json"
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")


def delete_profile(profile_id: str) -> bool:
    """Delete a profile. Returns True if it existed."""
    path = PROFILES_DIR / f"{profile_id}.json"
    if not path.exists():
        return False
    path.unlink()
    # Clear active marker if this was the active profile
    if get_active_profile_id() == profile_id:
        ACTIVE_FILE.unlink(missing_ok=True)
    return True


def get_active_profile_id() -> str | None:
    """Return the ID of the last-active profile, or None."""
    if not ACTIVE_FILE.exists():
        return None
    return ACTIVE_FILE.read_text(encoding="utf-8").strip() or None


def set_active(profile_id: str) -> None:
    """Mark a profile as the active one."""
    _ensure_dir()
    ACTIVE_FILE.write_text(profile_id, encoding="utf-8")


def get_last_profile() -> CameraProfile | None:
    """Load the last-active profile. Returns None if none set or file missing."""
    active_id = get_active_profile_id()
    if active_id is None:
        return None
    return load_profile(active_id)
