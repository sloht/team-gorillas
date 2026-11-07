"""Load the official tournament code from ./official and convert cards.

The official files import each other by their plain names (`from evaluator
import ...`), so this module puts ./official on sys.path once and re-exports
the three modules:

    from official_rules import game, evaluator, judge_engine

Card formats
------------
The team bots use (rank, suit) tuples: rank 6..14 (ace = 14), suit 0..3 with
letters "shdc". The official code uses ints 0..35: rank = card >> 2 (0 = six,
8 = ace), suit = card & 3 with letters "cdhs". Conversion keeps the printed
card the same ("As" stays "As"), so hand keys and tables are unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

OFFICIAL_DIR = Path(__file__).resolve().with_name("official")
if str(OFFICIAL_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DIR))

import evaluator  # noqa: E402  (official/evaluator.py)
import game  # noqa: E402  (official/game.py)
import judge_engine  # noqa: E402  (official/judge_engine.py)

TEAM_SUITS = "shdc"
OFFICIAL_COMMIT = "c83ab6e"  # heads-up-bundle commit the files in ./official come from


def to_official(card: tuple[int, int]) -> int:
    """(rank, suit) team card -> official card int."""
    rank, suit = card
    return ((rank - 6) << 2) | evaluator.SUIT_CHARS.index(TEAM_SUITS[suit])


def from_official(card: int) -> tuple[int, int]:
    """Official card int -> (rank, suit) team card."""
    return (card >> 2) + 6, TEAM_SUITS.index(evaluator.SUIT_CHARS[card & 3])


def cards_to_official(cards: Iterable[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(to_official(card) for card in cards)


def cards_from_official(cards: Iterable[int]) -> tuple[tuple[int, int], ...]:
    return tuple(from_official(card) for card in cards)


__all__ = ["OFFICIAL_COMMIT", "OFFICIAL_DIR","cards_from_official", "cards_to_official", "evaluator",
           "from_official", "game", "judge_engine", "to_official"]
