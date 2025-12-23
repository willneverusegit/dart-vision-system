"""
Player data structure and statistics.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from src.core import Hit


@dataclass
class Player:
    """Represents a player in the game."""
    name: str
    starting_score: int = 501

    # Current state
    current_score: int = field(init=False)
    darts_thrown: int = 0

    # Turn history
    turn_history: List[List[Hit]] = field(default_factory=list)

    # Statistics
    total_score: int = 0  # Total points scored
    highest_turn: int = 0
    checkout_attempts: int = 0
    checkouts: int = 0

    def __post_init__(self):
        """Initialize current score."""
        self.current_score = self.starting_score

    def add_hit(self, hit: Hit) -> bool:
        """
        Add a hit to current turn.

        Args:
            hit: Hit to add

        Returns:
            True if turn should continue, False if complete
        """
        # Get current turn (create if needed)
        if not self.turn_history or len(self.turn_history[-1]) >= 3:
            self.turn_history.append([])

        current_turn = self.turn_history[-1]
        current_turn.append(hit)

        self.darts_thrown += 1

        # Check if turn is complete (3 darts)
        return len(current_turn) < 3

    def get_current_turn(self) -> List[Hit]:
        """Get hits in current turn."""
        if not self.turn_history or len(self.turn_history[-1]) >= 3:
            return []
        return self.turn_history[-1]

    def get_current_turn_score(self) -> int:
        """Get total score of current turn."""
        return sum(hit.score for hit in self.get_current_turn())

    def undo_last_hit(self) -> Optional[Hit]:
        """
        Undo last hit.

        Returns:
            The removed hit, or None if no hits to undo
        """
        if not self.turn_history:
            return None

        current_turn = self.turn_history[-1]

        if not current_turn:
            # Current turn is empty, remove it and go to previous
            if len(self.turn_history) > 1:
                self.turn_history.pop()
                current_turn = self.turn_history[-1]
            else:
                return None

        if current_turn:
            hit = current_turn.pop()
            self.darts_thrown -= 1
            return hit

        return None

    def finalize_turn(self, new_score: int) -> int:
        """
        Finalize current turn and update score.

        Args:
            new_score: New score after turn

        Returns:
            Points scored this turn
        """
        turn_score = self.get_current_turn_score()

        # Update score
        old_score = self.current_score
        self.current_score = new_score

        # Update statistics
        self.total_score += turn_score
        if turn_score > self.highest_turn:
            self.highest_turn = turn_score

        return turn_score

    def reset(self) -> None:
        """Reset player to starting state."""
        self.current_score = self.starting_score
        self.darts_thrown = 0
        self.turn_history.clear()
        self.total_score = 0
        self.highest_turn = 0
        self.checkout_attempts = 0
        self.checkouts = 0

    @property
    def average_per_dart(self) -> float:
        """Calculate average score per dart."""
        if self.darts_thrown == 0:
            return 0.0
        return self.total_score / self.darts_thrown

    @property
    def average_per_turn(self) -> float:
        """Calculate average score per turn (3 darts)."""
        if not self.turn_history:
            return 0.0

        completed_turns = [t for t in self.turn_history if len(t) == 3]
        if not completed_turns:
            return 0.0

        total = sum(sum(hit.score for hit in turn) for turn in completed_turns)
        return total / len(completed_turns)