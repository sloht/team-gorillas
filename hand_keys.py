"""Card text and starting-hand keys shared by the strategy search and SuperBot.

This module is deliberately tiny and has no simulation code, so playing against
SuperBot never loads hand_strategy_search.py.
"""

from __future__ import annotations

from itertools import permutations
from typing import Iterable, Sequence

from short_deck_plo import Card

RANK_TEXT = {6: "6", 7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
TEXT_RANK = {text: rank for rank, text in RANK_TEXT.items()}
SUIT_TEXT = "shdc"
BEST_JSON = "best_strategy_by_hand.json"
_SUIT_PERMUTATIONS = tuple(permutations(range(4)))


def format_card(card: Card) -> str:
    return RANK_TEXT[card[0]] + SUIT_TEXT[card[1]]


def format_hand(hand: Sequence[Card]) -> str:
    return " ".join(format_card(card) for card in sorted(hand, reverse=True))


def parse_card(text: str) -> Card:
    if len(text) != 2 or text[0].upper() not in TEXT_RANK or text[1].lower() not in SUIT_TEXT:
        raise ValueError(f"{text!r} is not a Short Deck card (6-A, s/h/d/c).")
    return TEXT_RANK[text[0].upper()], SUIT_TEXT.index(text[1].lower())


def parse_hand(text: str) -> tuple[Card, ...]:
    """Parse cards such as "As Kh Qd Jc Ts" (spaces and commas optional)."""
    compact = text.replace(",", "").replace(" ", "")
    if len(compact) != 10:
        raise ValueError(f"Expected five cards such as 'As Kh Qd Jc Ts', got {text!r}.")
    cards = tuple(parse_card(compact[index:index + 2]) for index in range(0, 10, 2))
    if len(set(cards)) != 5:
        raise ValueError("The five cards must be different.")
    return cards


def canonical_hand(hand: Iterable[Card]) -> tuple[Card, ...]:
    """Return the representative of a hand's suit-symmetry class."""
    return min(
        tuple(sorted((rank, mapping[suit]) for rank, suit in hand))
        for mapping in _SUIT_PERMUTATIONS
    )


def hand_key(hand: Sequence[Card]) -> str:
    return format_hand(hand)


__all__ = [
    "BEST_JSON", "RANK_TEXT", "SUIT_TEXT", "TEXT_RANK", "canonical_hand",
    "format_card", "format_hand", "hand_key", "parse_card", "parse_hand",
]
