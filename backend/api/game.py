from fastapi import APIRouter, HTTPException

from backend.models.game import (
    GameConfig,
    GameResult,
    GameState,
    HitEvent,
)
from backend.scoring.game_engine import create_game, finish_game, process_throw

router = APIRouter(prefix="/api/game", tags=["game"])

# In-memory game state (single game for MVP)
_current_game: GameState | None = None


def _get_game() -> GameState:
    if _current_game is None or not _current_game.active:
        raise HTTPException(status_code=404, detail="No active game")
    return _current_game


@router.post("/start")
async def start_game(config: GameConfig) -> GameState:
    global _current_game
    if _current_game and _current_game.active:
        raise HTTPException(status_code=409, detail="Game already active")

    try:
        _current_game = create_game(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _current_game


@router.post("/stop")
async def stop_game() -> GameResult:
    global _current_game
    game = _get_game()
    result = finish_game(game)
    _current_game = None
    return result


@router.get("/state")
async def get_game_state() -> GameState:
    return _get_game()


@router.post("/throw")
async def register_throw(hit: HitEvent) -> GameState:
    game = _get_game()
    try:
        process_throw(game, hit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return game
