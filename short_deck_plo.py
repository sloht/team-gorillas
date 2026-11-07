"""The interface the team's bots are written against: cards, Action and Observation.

This module no longer referees anything. Every hand is played by the official
tournament engine (official/game.py) through official_match.py, which turns
the official observation into the Observation below and maps the bot's Action
onto the four official actions. Hand ranking here also delegates to the
official evaluator, so three of a kind beats a straight and a flush beats a
full house, exactly as in the tournament.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from official_rules import cards_to_official, evaluator

Card = tuple[int, int]  # (rank, suit); ranks are 6..14, suits are 0..3 ("shdc")
DECK: tuple[Card, ...] = tuple((rank, suit) for suit in range(4) for rank in range(6, 15))


def rank_five(cards: Sequence[Card]) -> int:
    """Official score of exactly five cards; bigger is better, equal ties."""
    if len(cards) != 5:
        raise ValueError("A five-card hand is required.")
    return evaluator.score_five(cards_to_official(cards))


def omaha_rank(hole: Sequence[Card], board: Sequence[Card]) -> int:
    """Official score using exactly two hole and three board cards (3 to 5 board cards)."""
    if len(hole) != 5:
        raise ValueError("Five-card Omaha requires exactly five hole cards.")
    if not 3 <= len(board) <= 5:
        raise ValueError("The board must contain between three and five cards.")
    table = evaluator.get_table()
    hole_bits = [1 << card for card in cards_to_official(hole)]
    board_bits = [1 << card for card in cards_to_official(board)]
    return max(
        table[hole_bits[i] | hole_bits[j] | sum(board_bits[k] for k in triple)]
        for i, j in combinations(range(5), 2)
        for triple in combinations(range(len(board)), 3)
    )


@dataclass(frozen=True)
class Action:
    """For bet/raise, amount is the target street contribution, not the increment.

    The official game only has two raise sizes. official_match.py plays a
    bet/raise as whichever of half-pot or pot lands closer to `amount`.
    """

    kind: str
    amount: int = 0

    @classmethod
    def fold(cls) -> "Action":
        return cls("fold")

    @classmethod
    def check(cls) -> "Action":
        return cls("check")

    @classmethod
    def call(cls) -> "Action":
        return cls("call")

    @classmethod
    def bet_to(cls, amount: int) -> "Action":
        return cls("bet", amount)

    @classmethod
    def raise_to(cls, amount: int) -> "Action":
        return cls("raise", amount)


@dataclass(frozen=True)
class Observation:
    """What a bot sees. official_match.to_bot_observation builds it from the official one.

    min_raise_to is the official half-pot raise total and max_raise_to the pot
    raise total, so a bot can only aim between the two sizes it may play.
    """

    seat: int
    button: int
    hole: tuple[Card, ...]
    board: tuple[Card, ...]
    pot: int
    stack: int
    street_contribution: int
    current_bet: int
    call_amount: int
    min_raise_to: int
    max_raise_to: int
    can_raise: bool
    big_blind: int
    opponents: int

    @property
    def preflop(self) -> bool:
        return not self.board

    @property
    def in_position(self) -> bool:
        """True on the button. Officially the button acts first preflop and last after it."""
        return self.seat == self.button


__all__ = ["Action", "Card", "DECK", "Observation", "omaha_rank", "rank_five"]
