from fastapi import APIRouter, HTTPException

from backend.models.game import (
    GameConfig,
    GameMode,
    GameResult,
    GameState,
    HitEvent,
    PlayerState,
)

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

    starting = config.starting_points if config.mode != GameMode.FREE else 0
    players = [
        PlayerState(name=name, remaining_points=starting) for name in config.player_names
    ]
    _current_game = GameState(
        mode=config.mode,
        players=players,
        active=True,
    )
    return _current_game


@router.post("/stop")
async def stop_game() -> GameResult:
    global _current_game
    game = _get_game()
    game.active = False

    winner = None
    for p in game.players:
        if p.remaining_points == 0 and game.mode != GameMode.FREE:
            winner = p.name
            break

    result = GameResult(
        winner=winner,
        players=game.players,
        total_rounds=sum(p.total_throws for p in game.players) // 3,
        history=game.history,
    )
    _current_game = None
    return result


@router.get("/state")
async def get_game_state() -> GameState:
    return _get_game()


@router.post("/throw")
async def register_throw(hit: HitEvent) -> GameState:
    game = _get_game()
    player = game.players[game.current_player]

    player.throws_this_turn.append(hit)
    player.total_throws += 1
    game.history.append(hit)

    if game.mode != GameMode.FREE:
        player.remaining_points -= hit.score
        if player.remaining_points < 0:
            # Bust: revert this turn
            for t in player.throws_this_turn:
                player.remaining_points += t.score
            player.throws_this_turn = []
            # Advance to next player
            game.current_player = (game.current_player + 1) % len(game.players)
            return game

    # After 3 throws, advance to next player
    if len(player.throws_this_turn) >= 3:
        player.throws_this_turn = []
        game.current_player = (game.current_player + 1) % len(game.players)

    return game
