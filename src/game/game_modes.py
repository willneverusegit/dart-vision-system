"""
Game modes (501, Cricket, etc.) with rule implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from src.core import Hit
from .player import Player


class GameMode(ABC):
    """Abstract base class for game modes."""

    @abstractmethod
    def get_name(self) -> str:
        """Get game mode name."""
        pass

    @abstractmethod
    def get_starting_score(self) -> Optional[int]:
        """Get starting score (None for non-countdown games)."""
        pass

    @abstractmethod
    def process_turn(
            self,
            player: Player,
            hits: List[Hit]
    ) -> Tuple[bool, Optional[str]]:
        """
        Process a completed turn.

        Args:
            player: Current player
            hits: Hits in this turn

        Returns:
            (is_valid, message) where:
                - is_valid: Whether turn is valid
                - message: Optional message (e.g., "BUST!" or "WINNER!")
        """
        pass

    @abstractmethod
    def check_winner(self, player: Player) -> bool:
        """Check if player has won."""
        pass


class Mode501(GameMode):
    """
    501 game mode with configurable checkout rules.

    Rules:
    - Start at 501 points
    - Subtract each hit from score
    - Must finish exactly on 0
    - Double-out: Must finish with a double
    - Bust if: score goes below 0, or below 2 (can't checkout)
    """

    def __init__(
            self,
            starting_score: int = 501,
            double_out: bool = True,
            double_in: bool = False
    ):
        """
        Initialize 501 game mode.

        Args:
            starting_score: Starting score (501, 301, etc.)
            double_out: Require double to finish
            double_in: Require double to start scoring
        """
        self.starting_score = starting_score
        self.double_out = double_out
        self.double_in = double_in

    def get_name(self) -> str:
        """Get game mode name."""
        suffix = ""
        if self.double_out:
            suffix = " (Double Out)"
        if self.double_in:
            suffix += " (Double In)"
        return f"{self.starting_score}{suffix}"

    def get_starting_score(self) -> int:
        """Get starting score."""
        return self.starting_score

    def process_turn(
            self,
            player: Player,
            hits: List[Hit]
    ) -> Tuple[bool, Optional[str]]:
        """Process turn for 501 game."""
        turn_score = sum(hit.score for hit in hits)
        new_score = player.current_score - turn_score

        # Check for bust conditions
        if new_score < 0:
            return False, "BUST! (Score below 0)"

        if new_score == 0:
            # Potential checkout
            if self.double_out:
                last_hit = hits[-1] if hits else None
                if last_hit and last_hit.multiplier == 2:
                    player.finalize_turn(0)
                    player.checkouts += 1
                    return True, "CHECKOUT! 🎯"
                else:
                    return False, "BUST! (Must finish on double)"
            else:
                player.finalize_turn(0)
                return True, "CHECKOUT! 🎯"

        if new_score == 1 and self.double_out:
            return False, "BUST! (Cannot checkout on 1)"

        # Valid turn
        player.finalize_turn(new_score)

        # Check if in checkout range
        if new_score <= 170 and new_score > 1:
            return True, "In checkout range!"

        return True, None

    def check_winner(self, player: Player) -> bool:
        """Check if player has won."""
        return player.current_score == 0


class ModeCricket(GameMode):
    """
    Cricket game mode.

    Rules:
    - Close numbers 20, 19, 18, 17, 16, 15, and Bulls
    - Need 3 hits to close a number
    - After closing, score points on open numbers
    - First to close all numbers with most points wins
    """

    CRICKET_NUMBERS = [20, 19, 18, 17, 16, 15, 25]  # 25 = Bull

    def __init__(self):
        """Initialize Cricket mode."""
        # Track hits per number per player (stored externally)
        pass

    def get_name(self) -> str:
        """Get game mode name."""
        return "Cricket"

    def get_starting_score(self) -> Optional[int]:
        """Cricket doesn't use countdown scoring."""
        return None

    def process_turn(
            self,
            player: Player,
            hits: List[Hit]
    ) -> Tuple[bool, Optional[str]]:
        """Process turn for Cricket."""
        # TODO: Implement Cricket logic
        # This would require additional state tracking
        return True, "Cricket mode (coming soon)"

    def check_winner(self, player: Player) -> bool:
        """Check if player has won Cricket."""
        # TODO: Implement Cricket win condition
        return False


class ModeTraining(GameMode):
    """
    Training mode - no rules, just practice.

    Tracks all hits for analysis.
    """

    def get_name(self) -> str:
        """Get game mode name."""
        return "Training"

    def get_starting_score(self) -> Optional[int]:
        """Training has no score."""
        return None

    def process_turn(
            self,
            player: Player,
            hits: List[Hit]
    ) -> Tuple[bool, Optional[str]]:
        """Process turn for training."""
        turn_score = sum(hit.score for hit in hits)
        player.total_score += turn_score
        player.darts_thrown += len(hits)

        return True, None

    def check_winner(self, player: Player) -> bool:
        """Training has no winner."""
        return False