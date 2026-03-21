from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class GameMode(StrEnum):
    FREE = "free"
    X301 = "301"
    X501 = "501"
    CRICKET = "cricket"


class DisplayMode(StrEnum):
    ECO = "eco"
    NORMAL = "normal"
    DEBUG = "debug"


class HitEvent(BaseModel):
    """Event generated from dart detection."""

    timestamp: datetime = Field(default_factory=datetime.now)
    camera_ids: list[str] = []
    image_points: dict[str, tuple[float, float]] = {}  # camera_id -> (x, y)
    world_point: tuple[float, float, float] | None = None  # 3D point
    board_point: tuple[float, float] | None = None  # 2D board coordinates
    field: str = ""  # e.g. "T20", "D16", "BULL"
    score: int = 0
    multiplier: int = 1
    confidence: float = 1.0


class ThrowResult(StrEnum):
    """Result of processing a throw."""

    OK = "ok"
    BUST = "bust"
    CHECKOUT = "checkout"


class CricketMarks(BaseModel):
    """Cricket marks per number for a player."""

    marks: dict[int, int] = Field(
        default_factory=lambda: {n: 0 for n in [15, 16, 17, 18, 19, 20, 25]}
    )

    def is_closed(self, number: int) -> bool:
        return self.marks.get(number, 0) >= 3

    def all_closed(self) -> bool:
        return all(m >= 3 for m in self.marks.values())


class PlayerState(BaseModel):
    """State of a single player."""

    name: str
    remaining_points: int = 0
    throws_this_turn: list[HitEvent] = []
    total_throws: int = 0
    cricket: CricketMarks = Field(default_factory=CricketMarks)
    cricket_points: int = 0


class GameConfig(BaseModel):
    """Configuration to start a new game."""

    mode: GameMode = GameMode.FREE
    player_names: list[str] = ["Player 1"]
    starting_points: int = 501


class GameState(BaseModel):
    """Full game state."""

    mode: GameMode = GameMode.FREE
    display_mode: DisplayMode = DisplayMode.NORMAL
    players: list[PlayerState] = []
    current_player: int = 0
    current_round: int = 1
    history: list[HitEvent] = []
    active: bool = False
    winner: str | None = None
    last_throw_result: ThrowResult = ThrowResult.OK


class GameResult(BaseModel):
    """Result when a game ends."""

    winner: str | None = None
    players: list[PlayerState] = []
    total_rounds: int = 0
    history: list[HitEvent] = []
