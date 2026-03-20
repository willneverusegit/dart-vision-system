import json

from backend.models import (
    BoardModel,
    CameraInfo,
    CameraProfile,
    CameraRole,
    GameConfig,
    GameResult,
    GameState,
    HitEvent,
    StereoProfile,
)


def test_camera_info_serialization():
    cam = CameraInfo(id="cam0", name="USB Camera")
    data = json.loads(cam.model_dump_json())
    assert data["id"] == "cam0"
    assert data["available"] is True


def test_camera_profile_roundtrip():
    profile = CameraProfile(id="cam0", role=CameraRole.LEFT, resolution=(640, 480))
    data = json.loads(profile.model_dump_json())
    restored = CameraProfile.model_validate(data)
    assert restored.id == "cam0"
    assert restored.role == CameraRole.LEFT
    assert restored.resolution == (640, 480)
    assert restored.intrinsics is None


def test_board_model_defaults():
    board = BoardModel(homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert board.board_radius == 170.0
    assert len(board.sector_angles) == 20
    assert board.ring_radii["bull"] == 6.35


def test_hit_event_defaults():
    hit = HitEvent(score=60, field="T20", multiplier=3)
    assert hit.score == 60
    assert hit.confidence == 1.0
    assert hit.camera_ids == []


def test_game_state_serialization():
    state = GameState(active=True)
    data = json.loads(state.model_dump_json())
    restored = GameState.model_validate(data)
    assert restored.active is True
    assert restored.current_player == 0


def test_game_config_defaults():
    config = GameConfig()
    assert config.mode == "free"
    assert config.player_names == ["Player 1"]
    assert config.starting_points == 501


def test_game_result():
    result = GameResult(winner="Alice", total_rounds=10)
    assert result.winner == "Alice"


def test_stereo_profile():
    sp = StereoProfile(
        camera_left_id="cam0",
        camera_right_id="cam1",
        rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        translation_vector=[0.1, 0, 0],
        reprojection_error=0.3,
    )
    data = json.loads(sp.model_dump_json())
    assert data["reprojection_error"] == 0.3
