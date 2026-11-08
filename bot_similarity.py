"""Behaviour fingerprints, for dropping bots that play (almost) the same.

A fingerprint is the action a bot chooses in each of a fixed set of game
situations (FINGERPRINT_SIZE observations from hands between varied bots,
the same in every process). Two bots agree on a share of those situations;
dedupe() keeps a bot only if it agrees with every bot already kept on less
than the threshold. Fingerprints never change for a bot, so they are cached
in <folder>/fingerprints.json.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from official_match import as_seat
from official_rules import game

FINGERPRINT_SIZE = 300
_SOURCE_SEED = "fingerprint-v1"


@lru_cache(maxsize=1)
def fingerprint_observations() -> tuple:
    """A fixed, varied set of decision points (identical in every process)."""
    from diverse_bots import DIVERSE_BOTS
    from run_tournament import ALL_BUILTIN_STRATEGIES
    makers = dict(ALL_BUILTIN_STRATEGIES)
    pool = [makers[n] for n in ("Random", "Maniac", "CallingStation", "TightAggressive", "PotShove08",
                                "Rock", "LooseAggressive", "AggressiveValue08")]
    pool += [DIVERSE_BOTS[n] for n in ("RuleLoose", "MixedStrategy", "Exploiter", "Floater")]
    rng = random.Random(_SOURCE_SEED)
    observations = []
    hand = 0
    while len(observations) < FINGERPRINT_SIZE * 4:
        players = [as_seat(rng.choice(pool)()), as_seat(rng.choice(pool)())]
        for seat, player in enumerate(players):
            player.set_seed(hand * 2 + seat)
        table = game.Game(game.make_deal(_SOURCE_SEED, hand), hand % 2, hand_index=hand)
        while not table.is_terminal:
            obs = table.observation(table.to_act)
            observations.append(obs)
            table.apply(players[table.to_act].act(obs))
        hand += 1
    rng.shuffle(observations)
    by_street = {s: [o for o in observations if o.street == s] for s in game.Street}
    picked = []
    for street in game.Street:  # a quarter from each street, so later streets count too
        picked += by_street[street][:FINGERPRINT_SIZE // 4]
    return tuple(picked)


def fingerprint(player) -> str:
    """The player's action (0-3) in each fingerprint situation, as a string."""
    seat = as_seat(player)
    seat.set_seed(12345)
    return "".join(str(int(seat.act(obs))) for obs in fingerprint_observations())


def agreement(a: str, b: str) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def dedupe(order: Iterable[str], prints: dict[str, str], threshold: float,
           keep_always: set[str] = frozenset()) -> tuple[list[str], dict[str, str]]:
    """Keep bots in `order` unless one already kept agrees with them >= threshold.
    Returns (kept ids, {dropped id: the kept bot it duplicates})."""
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for bot_id in order:
        twin = None if bot_id in keep_always else next(
            (k for k in kept if agreement(prints[bot_id], prints[k]) >= threshold), None)
        if twin is None:
            kept.append(bot_id)
        else:
            dropped[bot_id] = twin
    return kept, dropped


class FingerprintCache:
    """Fingerprints by bot id, computed on demand (in a worker pool) and saved."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.prints: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}

    def ensure(self, specs: dict[str, dict], work: Callable, pool=None) -> dict[str, str]:
        missing = [spec for bot_id, spec in specs.items() if bot_id not in self.prints]
        if missing:
            results = pool.map(work, missing, chunksize=4) if pool else map(work, missing)
            for bot_id, print_ in results:
                self.prints[bot_id] = print_
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.prints))
            tmp.replace(self.path)
        return self.prints
