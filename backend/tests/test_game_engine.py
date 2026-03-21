"""Tests for game engine: player management, X01 rules, Cricket mode."""

import pytest

from backend.models.game import (
    GameConfig,
    GameMode,
    HitEvent,
    ThrowResult,
)
from backend.scoring.game_engine import (
    THROWS_PER_TURN,
    create_game,
    finish_game,
    process_throw,
)


def _hit(field: str, score: int, multiplier: int = 1) -> HitEvent:
    """Helper to create a HitEvent."""
    return HitEvent(field=field, score=score, multiplier=multiplier)


# --- Player Management (P6-T1) ---


class TestPlayerManagement:
    def test_create_game_single_player(self):
        game = create_game(GameConfig(mode=GameMode.FREE, player_names=["Alice"]))
        assert len(game.players) == 1
        assert game.players[0].name == "Alice"
        assert game.active is True

    def test_create_game_max_players(self):
        names = [f"P{i}" for i in range(8)]
        game = create_game(GameConfig(player_names=names))
        assert len(game.players) == 8

    def test_create_game_too_many_players(self):
        names = [f"P{i}" for i in range(9)]
        with pytest.raises(ValueError, match="1-8"):
            create_game(GameConfig(player_names=names))

    def test_create_game_no_players(self):
        with pytest.raises(ValueError, match="1-8"):
            create_game(GameConfig(player_names=[]))

    def test_create_game_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            create_game(GameConfig(player_names=["Alice", "Alice"]))

    def test_turn_advances_after_3_throws(self):
        game = create_game(GameConfig(mode=GameMode.FREE, player_names=["Alice", "Bob"]))
        assert game.current_player == 0
        for _ in range(THROWS_PER_TURN):
            process_throw(game, _hit("S20", 20))
        assert game.current_player == 1

    def test_turn_cycles_back(self):
        game = create_game(GameConfig(mode=GameMode.FREE, player_names=["Alice", "Bob"]))
        # Alice throws 3
        for _ in range(3):
            process_throw(game, _hit("S1", 1))
        # Bob throws 3
        for _ in range(3):
            process_throw(game, _hit("S1", 1))
        # Back to Alice
        assert game.current_player == 0
        assert game.current_round == 2

    def test_player_total_throws_counted(self):
        game = create_game(GameConfig(mode=GameMode.FREE, player_names=["Alice"]))
        for _ in range(3):
            process_throw(game, _hit("S20", 20))
        assert game.players[0].total_throws == 3

    def test_throw_on_inactive_game(self):
        game = create_game(GameConfig(mode=GameMode.FREE, player_names=["Alice"]))
        game.active = False
        with pytest.raises(ValueError, match="not active"):
            process_throw(game, _hit("S20", 20))


# --- X01 Rules (P6-T2) ---


class TestX01Rules:
    def test_501_starting_points(self):
        game = create_game(
            GameConfig(mode=GameMode.X501, player_names=["Alice"], starting_points=501)
        )
        assert game.players[0].remaining_points == 501

    def test_301_starting_points(self):
        game = create_game(
            GameConfig(mode=GameMode.X301, player_names=["Alice"], starting_points=301)
        )
        assert game.players[0].remaining_points == 301

    def test_normal_scoring_deducts(self):
        game = create_game(
            GameConfig(mode=GameMode.X501, player_names=["Alice"], starting_points=501)
        )
        result = process_throw(game, _hit("T20", 20, 3))
        assert result == ThrowResult.OK
        assert game.players[0].remaining_points == 441  # 501 - 60

    def test_bust_on_overshoot(self):
        game = create_game(
            GameConfig(mode=GameMode.X301, player_names=["Alice", "Bob"], starting_points=10)
        )
        result = process_throw(game, _hit("S20", 20))
        assert result == ThrowResult.BUST
        assert game.players[0].remaining_points == 10  # reverted

    def test_bust_on_remaining_one(self):
        """Remaining=1 is bust because you can't finish with a double."""
        game = create_game(
            GameConfig(mode=GameMode.X301, player_names=["Alice", "Bob"], starting_points=3)
        )
        result = process_throw(game, _hit("S2", 2))
        assert result == ThrowResult.BUST
        assert game.players[0].remaining_points == 3

    def test_bust_on_zero_without_double(self):
        """Reaching 0 with a single is bust (double-out required)."""
        game = create_game(
            GameConfig(mode=GameMode.X301, player_names=["Alice", "Bob"], starting_points=20)
        )
        result = process_throw(game, _hit("S20", 20))
        assert result == ThrowResult.BUST
        assert game.players[0].remaining_points == 20

    def test_checkout_with_double(self):
        game = create_game(
            GameConfig(mode=GameMode.X301, player_names=["Alice", "Bob"], starting_points=40)
        )
        result = process_throw(game, _hit("D20", 20, 2))
        assert result == ThrowResult.CHECKOUT
        assert game.players[0].remaining_points == 0
        assert game.winner == "Alice"
        assert game.active is False

    def test_checkout_with_bull(self):
        """BULL (50) counts as a valid checkout."""
        game = create_game(
            GameConfig(mode=GameMode.X301, player_names=["Alice"], starting_points=50)
        )
        result = process_throw(game, _hit("BULL", 50, 1))
        assert result == ThrowResult.CHECKOUT
        assert game.winner == "Alice"

    def test_bust_reverts_all_turn_throws(self):
        """On bust, ALL throws this turn are reverted."""
        game = create_game(
            GameConfig(mode=GameMode.X301, player_names=["Alice", "Bob"], starting_points=100)
        )
        process_throw(game, _hit("T20", 20, 3))  # 100 - 60 = 40
        assert game.players[0].remaining_points == 40
        result = process_throw(game, _hit("S20", 20))  # 40 - 20 = 20, but...
        assert game.players[0].remaining_points == 20
        # Third throw busts
        result = process_throw(game, _hit("T20", 20, 3))  # 20 - 60 < 0 = bust
        assert result == ThrowResult.BUST
        assert game.players[0].remaining_points == 100  # all reverted

    def test_turn_advances_on_bust(self):
        game = create_game(
            GameConfig(mode=GameMode.X301, player_names=["Alice", "Bob"], starting_points=10)
        )
        process_throw(game, _hit("S20", 20))  # bust
        assert game.current_player == 1  # Bob's turn

    def test_finish_game_result(self):
        game = create_game(
            GameConfig(mode=GameMode.X501, player_names=["Alice", "Bob"], starting_points=40)
        )
        process_throw(game, _hit("D20", 20, 2))  # Alice checkout
        result = finish_game(game)
        assert result.winner == "Alice"
        assert result.total_rounds == 1


# --- Cricket Mode (P6-T3) ---


class TestCricketMode:
    def _cricket_game(self, names: list[str] | None = None):
        if names is None:
            names = ["Alice", "Bob"]
        return create_game(GameConfig(mode=GameMode.CRICKET, player_names=names))

    def test_single_marks_one(self):
        game = self._cricket_game()
        process_throw(game, _hit("S20", 20))
        assert game.players[0].cricket.marks[20] == 1

    def test_double_marks_two(self):
        game = self._cricket_game()
        process_throw(game, _hit("D20", 20, 2))
        assert game.players[0].cricket.marks[20] == 2

    def test_triple_closes(self):
        game = self._cricket_game()
        process_throw(game, _hit("T20", 20, 3))
        assert game.players[0].cricket.marks[20] == 3
        assert game.players[0].cricket.is_closed(20)

    def test_non_cricket_number_ignored(self):
        game = self._cricket_game()
        process_throw(game, _hit("S10", 10))
        assert game.players[0].cricket_points == 0

    def test_scoring_after_close_opponent_open(self):
        """Once closed, extra marks score if opponent hasn't closed."""
        game = self._cricket_game()
        # Alice closes 20 with a triple
        process_throw(game, _hit("T20", 20, 3))
        # Alice hits another S20 -> scores 20 (Bob still open on 20)
        process_throw(game, _hit("S20", 20))
        assert game.players[0].cricket_points == 20

    def test_no_scoring_if_all_opponents_closed(self):
        """No points scored if all opponents have closed the number."""
        game = self._cricket_game()
        # Alice closes 20
        process_throw(game, _hit("T20", 20, 3))
        process_throw(game, _hit("S1", 1))  # filler
        process_throw(game, _hit("S1", 1))  # filler, turn ends

        # Bob closes 20
        process_throw(game, _hit("T20", 20, 3))
        process_throw(game, _hit("S1", 1))
        process_throw(game, _hit("S1", 1))

        # Alice hits 20 again - no points because Bob also closed
        process_throw(game, _hit("S20", 20))
        assert game.players[0].cricket_points == 0

    def test_bull_counts_as_25(self):
        game = self._cricket_game()
        process_throw(game, _hit("BULL", 50, 1))
        assert game.players[0].cricket.marks[25] == 2  # Bull = 2 marks

    def test_outer_bull_marks_one(self):
        game = self._cricket_game()
        process_throw(game, _hit("25", 25, 1))
        assert game.players[0].cricket.marks[25] == 1

    def test_win_all_closed_highest_points(self):
        game = self._cricket_game()
        # Alice closes all numbers
        for n in [15, 16, 17, 18, 19, 20]:
            process_throw(game, _hit(f"T{n}", n, 3))
            # Fill remaining throws to advance turn (if needed)
            while game.current_player == 0 and len(game.players[0].throws_this_turn) < 3:
                process_throw(game, _hit("MISS", 0, 0))
            # Bob throws misses
            while game.current_player == 1:
                process_throw(game, _hit("MISS", 0, 0))

        # Now close bull (25) — need 3 marks total
        # BULL = 2 marks, 25 = 1 mark
        process_throw(game, _hit("BULL", 50, 1))  # 2 marks
        process_throw(game, _hit("25", 25, 1))  # 1 mark = 3 total -> closed, all closed

        assert game.winner == "Alice"
        assert game.active is False

    def test_no_win_if_opponent_has_more_points(self):
        """Can't win cricket if opponent has more points."""
        game = self._cricket_game()
        # Bob scores points first: Bob closes 20 and scores
        # Alice throws misses
        for _ in range(3):
            process_throw(game, _hit("MISS", 0, 0))
        # Bob closes 20
        process_throw(game, _hit("T20", 20, 3))
        # Bob scores 20 twice (Alice has 20 open)
        process_throw(game, _hit("S20", 20))
        process_throw(game, _hit("S20", 20))
        assert game.players[1].cricket_points == 40

        # Alice closes everything quickly but has 0 points
        for n in [15, 16, 17, 18, 19, 20]:
            process_throw(game, _hit(f"T{n}", n, 3))
            while game.current_player == 0 and len(game.players[0].throws_this_turn) < 3:
                process_throw(game, _hit("MISS", 0, 0))
            while game.current_player == 1:
                process_throw(game, _hit("MISS", 0, 0))

        # Alice closes bull
        process_throw(game, _hit("BULL", 50, 1))
        process_throw(game, _hit("25", 25, 1))

        # Alice has 0 points, Bob has 40 -> Alice can't win
        assert game.winner is None
        assert game.active is True


class TestFreeMode:
    def test_free_mode_accumulates(self):
        game = create_game(GameConfig(mode=GameMode.FREE, player_names=["Alice"]))
        process_throw(game, _hit("T20", 20, 3))
        assert game.players[0].remaining_points == 60

    def test_free_mode_no_bust(self):
        game = create_game(GameConfig(mode=GameMode.FREE, player_names=["Alice"]))
        for _ in range(3):
            process_throw(game, _hit("T20", 20, 3))
        assert game.players[0].remaining_points == 180
