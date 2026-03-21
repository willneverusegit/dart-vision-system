"""Game engine with player management, X01 rules, and Cricket mode."""

from backend.models.game import (
    GameConfig,
    GameMode,
    GameResult,
    GameState,
    HitEvent,
    PlayerState,
    ThrowResult,
)

MAX_PLAYERS = 8
THROWS_PER_TURN = 3
CRICKET_NUMBERS = {15, 16, 17, 18, 19, 20, 25}


def create_game(config: GameConfig) -> GameState:
    """Create a new game from config.

    Args:
        config: Game configuration with mode, player names, starting points.

    Returns:
        Initialized GameState.

    Raises:
        ValueError: If player count is invalid or names are not unique.
    """
    names = config.player_names
    if len(names) < 1 or len(names) > MAX_PLAYERS:
        raise ValueError(f"Player count must be 1-{MAX_PLAYERS}, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("Player names must be unique")

    if config.mode in (GameMode.X301, GameMode.X501):
        starting = config.starting_points
    else:
        starting = 0

    players = [PlayerState(name=name, remaining_points=starting) for name in names]
    return GameState(mode=config.mode, players=players, active=True)


def next_turn(game: GameState) -> None:
    """Advance to the next player's turn."""
    player = game.players[game.current_player]
    player.throws_this_turn = []
    game.last_throw_result = ThrowResult.OK

    game.current_player = (game.current_player + 1) % len(game.players)
    if game.current_player == 0:
        game.current_round += 1


def process_throw(game: GameState, hit: HitEvent) -> ThrowResult:
    """Process a throw for the current player.

    Dispatches to the mode-specific handler. Advances turn after 3 throws,
    bust, or checkout.

    Returns:
        ThrowResult indicating what happened (OK, BUST, CHECKOUT).
    """
    if not game.active:
        raise ValueError("Game is not active")
    if game.winner:
        raise ValueError("Game already has a winner")

    player = game.players[game.current_player]
    player.throws_this_turn.append(hit)
    player.total_throws += 1
    game.history.append(hit)

    if game.mode == GameMode.FREE:
        result = _process_free(player, hit)
    elif game.mode in (GameMode.X301, GameMode.X501):
        result = _process_x01(game, player, hit)
    elif game.mode == GameMode.CRICKET:
        result = _process_cricket(game, player, hit)
    else:
        result = ThrowResult.OK

    game.last_throw_result = result

    # Auto-advance turn on bust, checkout, or 3 throws
    if result in (ThrowResult.BUST, ThrowResult.CHECKOUT):
        next_turn(game)
        if result == ThrowResult.CHECKOUT:
            game.active = False
    elif len(player.throws_this_turn) >= THROWS_PER_TURN:
        next_turn(game)

    return result


def _process_free(player: PlayerState, hit: HitEvent) -> ThrowResult:
    """Free mode: just accumulate points."""
    player.remaining_points += hit.score * hit.multiplier
    return ThrowResult.OK


def _process_x01(game: GameState, player: PlayerState, hit: HitEvent) -> ThrowResult:
    """X01 mode with Double-Out rule.

    Bust conditions:
    - Score would reduce remaining below 0
    - Score would leave remaining at exactly 1 (can't finish with double)
    - Score reaches 0 but the finishing throw is not a double
    """
    throw_value = hit.score * hit.multiplier
    new_remaining = player.remaining_points - throw_value

    if new_remaining < 0 or new_remaining == 1:
        _revert_turn_x01(player)
        return ThrowResult.BUST

    if new_remaining == 0:
        if hit.multiplier != 2 and hit.field != "BULL":
            _revert_turn_x01(player)
            return ThrowResult.BUST
        player.remaining_points = 0
        game.winner = player.name
        return ThrowResult.CHECKOUT

    player.remaining_points = new_remaining
    return ThrowResult.OK


def _revert_turn_x01(player: PlayerState) -> None:
    """Revert X01 scoring: restore remaining_points to start-of-turn value.

    Only prior throws (not the current bust throw) were deducted, so we
    add back all throws except the last one (the bust throw was never deducted).
    """
    for t in player.throws_this_turn[:-1]:
        player.remaining_points += t.score * t.multiplier


def _process_cricket(game: GameState, player: PlayerState, hit: HitEvent) -> ThrowResult:
    """Cricket mode scoring.

    Numbers 15-20 and Bull (25/BULL). Each number needs 3 marks to close.
    Triples count as 3 marks, doubles as 2. Once a player has closed a number,
    additional hits score points IF any opponent still has it open.
    Win condition: all numbers closed AND highest (or tied) points.
    """
    number = _cricket_number(hit.field)
    if number is None:
        return ThrowResult.OK  # Not a cricket number, no effect

    marks_to_add = hit.multiplier if hit.field not in ("BULL", "25") else 1
    if hit.field == "BULL":
        marks_to_add = 2  # Bull counts as 2 marks (double bull)
        number = 25

    current_marks = player.cricket.marks.get(number, 0)
    new_marks = current_marks + marks_to_add

    # Marks that count toward closing (up to 3)
    closing_marks = min(new_marks, 3) - min(current_marks, 3)
    # Excess marks that could score points
    scoring_marks = marks_to_add - closing_marks

    player.cricket.marks[number] = new_marks

    # Score points for excess marks if opponents haven't closed this number
    if scoring_marks > 0 and new_marks > 3:
        opponents_open = any(
            not p.cricket.is_closed(number)
            for i, p in enumerate(game.players)
            if i != game.current_player
        )
        if opponents_open:
            point_value = 25 if number == 25 else number
            player.cricket_points += scoring_marks * point_value

    # Check win condition: all closed AND highest points
    if player.cricket.all_closed():
        has_most_points = all(
            player.cricket_points >= p.cricket_points
            for i, p in enumerate(game.players)
            if i != game.current_player
        )
        if has_most_points:
            game.winner = player.name
            return ThrowResult.CHECKOUT

    return ThrowResult.OK


def _cricket_number(field: str) -> int | None:
    """Extract the cricket-relevant number from a field name."""
    if field == "BULL":
        return 25
    if field == "25":
        return 25
    if field == "MISS" or len(field) < 2:
        return None

    try:
        number = int(field[1:])
    except ValueError:
        return None

    return number if number in CRICKET_NUMBERS else None


def finish_game(game: GameState) -> GameResult:
    """End the game and produce a result."""
    game.active = False
    return GameResult(
        winner=game.winner,
        players=game.players,
        total_rounds=game.current_round,
        history=game.history,
    )
