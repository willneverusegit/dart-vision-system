"""
Game state management.
"""
from typing import List, Optional
from dataclasses import dataclass, field
import logging

from src.core import Hit
from .player import Player
from .game_modes import GameMode, Mode501

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Manages overall game state."""
    players: List[Player] = field(default_factory=list)
    game_mode: GameMode = field(default_factory=lambda: Mode501())

    current_player_idx: int = 0
    game_started: bool = False
    game_finished: bool = False
    winner: Optional[Player] = None

    def add_player(self, name: str) -> Player:
        """
        Add a player to the game.

        Args:
            name: Player name

        Returns:
            The created player
        """
        starting_score = self.game_mode.get_starting_score()
        player = Player(
            name=name,
            starting_score=starting_score if starting_score else 0
        )
        self.players.append(player)
        logger.info(f"Player added: {name}")
        return player

    def remove_player(self, player: Player) -> bool:
        """
        Remove a player from the game.

        Args:
            player: Player to remove

        Returns:
            True if removed, False if not found
        """
        if player in self.players:
            self.players.remove(player)
            logger.info(f"Player removed: {player.name}")
            return True
        return False

    def start_game(self) -> bool:
        """
        Start the game.

        Returns:
            True if started successfully, False otherwise
        """
        if len(self.players) == 0:
            logger.error("Cannot start game: No players")
            return False

        self.game_started = True
        self.game_finished = False
        self.winner = None
        self.current_player_idx = 0

        logger.info(f"Game started: {self.game_mode.get_name()} with {len(self.players)} players")
        return True

    def get_current_player(self) -> Optional[Player]:
        """Get the current player."""
        if not self.players or not self.game_started:
            return None
        return self.players[self.current_player_idx]

    def add_hit(self, hit: Hit) -> bool:
        """
        Add a hit for the current player.

        Args:
            hit: Hit to add

        Returns:
            True if turn continues, False if turn is complete
        """
        player = self.get_current_player()
        if not player:
            return False

        continue_turn = player.add_hit(hit)

        logger.debug(
            f"{player.name} hit: {hit.score} points "
            f"(Turn: {player.get_current_turn_score()})"
        )

        return continue_turn

    def finalize_turn(self) -> Optional[str]:
        """
        Finalize current player's turn and advance to next player.

        Returns:
            Optional message (e.g., "BUST!" or "CHECKOUT!")
        """
        player = self.get_current_player()
        if not player:
            return None

        hits = player.get_current_turn()

        if not hits:
            logger.warning(f"{player.name} has no hits to finalize")
            return None

        # Process turn through game mode
        is_valid, message = self.game_mode.process_turn(player, hits)

        if not is_valid:
            # Bust - revert turn
            logger.info(f"{player.name} busted: {message}")
            for _ in range(len(hits)):
                player.undo_last_hit()

        # Check for winner
        if self.game_mode.check_winner(player):
            self.game_finished = True
            self.winner = player
            logger.info(f"Game finished! Winner: {player.name}")
            return "WINNER!"

        # Advance to next player
        self.next_player()

        return message

    def next_player(self) -> None:
        """Advance to next player."""
        if not self.players:
            return

        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        logger.debug(f"Next player: {self.get_current_player().name}")

    def undo_last_hit(self) -> bool:
        """
        Undo last hit.

        Returns:
            True if successful, False otherwise
        """
        player = self.get_current_player()
        if not player:
            return False

        hit = player.undo_last_hit()
        if hit:
            logger.info(f"Undone: {hit.score} points")
            return True

        return False

    def reset_game(self) -> None:
        """Reset game to initial state."""
        for player in self.players:
            player.reset()

        self.current_player_idx = 0
        self.game_started = False
        self.game_finished = False
        self.winner = None

        logger.info("Game reset")