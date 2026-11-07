"""A bot that looks at its starting hand and plays the strategy that did best with it.

The lookup table comes from `hand_strategy_search.py merge`. At the start of each
hand, SuperBot finds its five hole cards in the table (suit symmetry is handled)
and hands every decision for that hand to the stored best strategy. Hands that
are missing from the table use a fallback strategy.

Use it in a round robin like any team bot:

    python3 run_tournament.py \\
        --bot Super=super_bot:SuperBot \\
        --bot ShortDeckPLO=poker_bots:ShortDeckPLOBot \\
        --hands 2000 --seed 42

By default the table is read from ./hand_strategy_results; set the
SUPER_BOT_TABLE environment variable to point somewhere else (a results folder
or the best_strategy_by_hand.json file itself).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

from hand_strategy_search import BEST_JSON, canonical_hand, hand_key
from poker_bots import PokerBot
from run_tournament import ALL_BUILTIN_STRATEGIES
from short_deck_plo import Action, Observation

DEFAULT_TABLE = "hand_strategy_results"
TABLE_ENV = "SUPER_BOT_TABLE"
_STRATEGIES = dict(ALL_BUILTIN_STRATEGIES)


@lru_cache(maxsize=8)
def load_strategy_table(path: str, require_clear_winner: bool = False) -> tuple[dict[str, str], bool]:
    """Return ({hand key: strategy name}, uses_suit_symmetry) from a merged results table.

    With require_clear_winner, hands whose best strategy was not clearly ahead of
    the runner-up are left out, so the bot uses its fallback for them.
    """
    table_path = Path(path)
    if table_path.is_dir():
        table_path = table_path / BEST_JSON
    if not table_path.exists():
        raise FileNotFoundError(
            f"No strategy table at {table_path}. Run hand_strategy_search.py run and merge "
            f"first, or set {TABLE_ENV} to the results folder."
        )
    data = json.loads(table_path.read_text())
    table = {
        key: result["best"]
        for key, result in data["hands"].items()
        if not require_clear_winner or result.get("clear_winner", True)
    }
    unknown = sorted(set(table.values()) - set(_STRATEGIES))
    if unknown:
        raise ValueError(f"The table names strategies this code does not have: {', '.join(unknown)}")
    return table, data.get("suit_symmetry", True)


class SuperBot(PokerBot):
    """Picks the strategy to play for each hand from its starting cards."""

    name = "SuperBot"

    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        table_path: Optional[str] = None,
        fallback: str = "ShortDeckPLO",
        require_clear_winner: bool = False,
    ) -> None:
        super().__init__(seed)
        if fallback not in _STRATEGIES:
            raise ValueError(f"Unknown fallback strategy {fallback!r}.")
        path = table_path or os.environ.get(TABLE_ENV, DEFAULT_TABLE)
        self.table, self.suit_symmetry = load_strategy_table(str(path), require_clear_winner)
        self.fallback = fallback
        self._bots: dict[str, PokerBot] = {}
        self._seed = seed
        self._hole: Optional[tuple] = None
        self._board_length = 0
        self.current_strategy: Optional[str] = None
        self.strategy_counts: Counter[str] = Counter()

    def set_seed(self, seed: int) -> None:
        super().set_seed(seed)
        self._seed = seed
        for offset, bot in enumerate(self._bots.values(), 1):
            bot.set_seed(seed + offset)

    def strategy_for(self, hole: tuple) -> str:
        """Return the strategy name the table assigns to these hole cards."""
        hand = canonical_hand(hole) if self.suit_symmetry else tuple(sorted(hole))
        return self.table.get(hand_key(hand), self.fallback)

    def _bot(self, name: str) -> PokerBot:
        if name not in self._bots:
            bot = _STRATEGIES[name]()
            if self._seed is not None and hasattr(bot, "set_seed"):
                bot.set_seed(self._seed + len(self._bots) + 1)
            self._bots[name] = bot
        return self._bots[name]

    def act(self, observation: Observation) -> Action:
        # A new hand starts when the hole cards change or the board resets.
        if observation.hole != self._hole or len(observation.board) < self._board_length:
            self._hole = observation.hole
            self.current_strategy = self.strategy_for(observation.hole)
            self.strategy_counts[self.current_strategy] += 1
        self._board_length = len(observation.board)
        return self._bot(self.current_strategy).act(observation)


__all__ = ["SuperBot", "load_strategy_table"]
