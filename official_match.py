"""Play the team's bots under the official tournament rules and scoring.

Everything that evaluates strategies goes through this module:

* Hands are refereed by the official engine, ``game.Game`` (80-chip stacks,
  blinds 1/2, four actions, at most 4 raises per street, official showdowns).
* Bots keep their own interface (short_deck_plo.Observation / Action).
  ``OfficialSeat`` translates in both directions:
    - official Observation -> bot Observation (cards converted; min_raise_to is
      the half-pot raise total and max_raise_to the pot raise total);
    - bot Action -> official Action: fold -> FOLD (CHECK_CALL when checking is
      free), check/call -> CHECK_CALL, bet/raise to X -> RAISE_HALF or
      RAISE_POT, whichever total is closer to X (half wins a tie);
      raising when no raise is legal -> CHECK_CALL.
* Matches are duplicate, as in the tournament: ``deals`` deals from
  ``game.make_deal(seed, d)``, played twice with seats swapped, button on seat
  ``d % 2``. The two halves are scored by ``judge_engine.combine_halves``, the
  judge's own function: mean big blinds per deal with a 95% interval;
  win = 1 point, chop = 0.5, loss = 0.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from dataclasses import dataclass
from hashlib import blake2b
from typing import Callable, Iterable, Optional, Sequence

from official_rules import cards_from_official, game, judge_engine
from short_deck_plo import Action as BotAction
from short_deck_plo import Observation as BotObservation

OfficialAction = game.Action
BIG_BLIND = game.BIG_BLIND
STARTING_STACK = game.STARTING_STACK
NUM_DEALS = game.NUM_DEALS


# --------------------------------------------------------------------------- #
# Adapter between the bots and the official game
# --------------------------------------------------------------------------- #

def to_bot_observation(obs: "game.Observation") -> BotObservation:
    """Describe an official observation in the terms the team's bots use."""
    can_raise = OfficialAction.RAISE_HALF in obs.legal_actions
    current_bet = max(obs.my_bet, obs.opp_bet)
    if can_raise:
        low = obs.raise_to[OfficialAction.RAISE_HALF]
        high = obs.raise_to[OfficialAction.RAISE_POT]
    else:
        low = high = current_bet
    return BotObservation(
        seat=0,
        button=0 if obs.is_button else 1,
        hole=cards_from_official(obs.hole_cards),
        board=cards_from_official(obs.board),
        pot=obs.pot,
        stack=obs.my_stack,
        street_contribution=obs.my_bet,
        current_bet=current_bet,
        call_amount=obs.to_call,
        min_raise_to=low,
        max_raise_to=high,
        can_raise=can_raise,
        big_blind=obs.big_blind,
        opponents=1,
    )


def to_official_action(action: object, obs: "game.Observation") -> "game.Action":
    """Map a bot's Action onto one of the official legal actions."""
    legal = obs.legal_actions
    kind = getattr(action, "kind", None)
    if kind == "fold":
        return OfficialAction.FOLD if OfficialAction.FOLD in legal else OfficialAction.CHECK_CALL
    if kind in ("bet", "raise") and OfficialAction.RAISE_HALF in legal:
        target = getattr(action, "amount", 0)
        half = obs.raise_to[OfficialAction.RAISE_HALF]
        pot = obs.raise_to[OfficialAction.RAISE_POT]
        return OfficialAction.RAISE_HALF if abs(target - half) <= abs(target - pot) \
            else OfficialAction.RAISE_POT
    return OfficialAction.CHECK_CALL


class OfficialSeat:
    """Wraps a team bot so the official game can ask it for actions."""

    def __init__(self, bot: object) -> None:
        self.bot = bot
        self.name = getattr(bot, "name", type(bot).__name__)

    def set_seed(self, seed: int) -> None:
        set_seed = getattr(self.bot, "set_seed", None)
        if callable(set_seed):
            set_seed(seed)

    def act(self, obs: "game.Observation") -> "game.Action":
        return to_official_action(self.bot.act(to_bot_observation(obs)), obs)

    def on_hand_end(self, result: "game.HandResult") -> None:
        on_hand_end = getattr(self.bot, "on_hand_end", None)
        if callable(on_hand_end):
            on_hand_end(result)


def as_seat(player: object) -> object:
    """Accept either a team bot or something that already plays official actions."""
    return player if isinstance(player, OfficialSeat) or getattr(player, "plays_official", False) \
        else OfficialSeat(player)


# --------------------------------------------------------------------------- #
# Hands and matches
# --------------------------------------------------------------------------- #

def play_hand(players: Sequence[object], deal: "game.Deal", button: int,
              hand_index: int = 0) -> "game.Game":
    """Play one official hand; players[seat].act(official obs) -> official Action.

    Returns the finished Game (use .result() for nets, .hand_result(seat) for
    one player's view).
    """
    hand = game.Game(deal, button, hand_index=hand_index)
    while not hand.is_terminal:
        seat = hand.to_act
        hand.apply(players[seat].act(hand.observation(seat)))
    # Like the tournament judge: tell each player how the hand ended (on_hand_end).
    for seat, player in enumerate(players):
        on_hand_end = getattr(player, "on_hand_end", None)
        if callable(on_hand_end):
            on_hand_end(hand.hand_result(seat))
    return hand


def _half_seed(seed: object, half: int, seat: int) -> int:
    digest = blake2b(f"{seed}|half{half}|seat{seat}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def play_half(first: object, second: object, *, seed: object, half: int,
              deals: int = NUM_DEALS) -> "judge_engine.HalfResult":
    """One official half: `first` is agent1 (seat 0 in half 1, seat 1 in half 2)."""
    first, second = as_seat(first), as_seat(second)
    match_seed = str(seed)
    first_seat = judge_engine.agent1_seat_for_half(half)
    seats = (first, second) if first_seat == 0 else (second, first)
    # Each half starts fresh in the tournament; reseed bot randomness to match.
    for seat, player in enumerate(seats):
        if hasattr(player, "set_seed"):
            player.set_seed(_half_seed(match_seed, half, seat))
    nets: tuple[list[int], list[int]] = ([], [])
    for d in range(deals):
        result = play_hand(seats, game.make_deal(match_seed, d),
                           game.button_for_deal(d), hand_index=d).result()
        nets[0].append(result.nets[0])
        nets[1].append(result.nets[1])
    return judge_engine.HalfResult(
        half=half, match_seed=match_seed, num_deals=deals, agent1_seat=first_seat,
        nets_agent1=nets[first_seat], nets_agent2=nets[1 - first_seat],
        stats_agent1=judge_engine.AgentStats(), stats_agent2=judge_engine.AgentStats(),
        agent1_name=first.name, agent2_name=second.name,
    )


def run_match(first: object, second: object, *, deals: int = NUM_DEALS,
              seed: object = 42) -> "judge_engine.MatchResult":
    """A full duplicate match, scored by the official judge's combine_halves().

    The result's verdict is AGENT1_WIN (first wins), AGENT2_WIN or CHOP;
    points is {"agent1": .., "agent2": ..}; mean_bb is first's bb per deal.
    """
    if deals < 1:
        raise ValueError("deals must be at least 1.")
    half1 = play_half(first, second, seed=seed, half=1, deals=deals)
    half2 = play_half(first, second, seed=seed, half=2, deals=deals)
    return judge_engine.combine_halves(half1, half2)


def match_record(left: str, right: str, left_bot: object, right_bot: object,
                 seed: object, deals: int) -> dict:
    """Play one duplicate match and return a small JSON-friendly record."""
    result = run_match(left_bot, right_bot, deals=deals, seed=seed)
    return {
        "left": left, "right": right, "seed": seed, "deals": deals,
        "verdict": result.verdict,
        "points": [result.points["agent1"], result.points["agent2"]],
        "mean_bb": round(result.mean_bb, 6),
        "ci": [round(result.ci[0], 6), round(result.ci[1], 6)],
    }


# --------------------------------------------------------------------------- #
# Round robins
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Standing:
    name: str
    matches: int
    points: float
    wins: int
    losses: int
    chops: int
    avg_bb_per_deal: float          # mean of this strategy's per-match bb/deal
    seed_points: tuple[float, ...] = ()
    seed_matches: tuple[int, ...] = ()

    @property
    def seed_point_rate_range(self) -> Optional[tuple[float, float]]:
        rates = [p / m for p, m in zip(self.seed_points, self.seed_matches) if m]
        return (min(rates), max(rates)) if rates else None


def standings_from_records(records: Iterable[dict], names: Sequence[str],
                           seeds: Sequence[object] = ()) -> list[Standing]:
    """Rank by tournament points, then average bb per deal, then name."""
    totals = {name: {"matches": 0, "points": 0.0, "wins": 0, "losses": 0, "chops": 0,
                     "bb": 0.0, "seed_points": {s: 0.0 for s in seeds},
                     "seed_matches": {s: 0 for s in seeds}} for name in names}
    for record in records:
        for side, name, sign in ((0, record["left"], 1), (1, record["right"], -1)):
            row = totals[name]
            points = record["points"][side]
            row["matches"] += 1
            row["points"] += points
            row["bb"] += sign * record["mean_bb"]
            if points == 1:
                row["wins"] += 1
            elif points == 0.5:
                row["chops"] += 1
            else:
                row["losses"] += 1
            if record["seed"] in row["seed_points"]:
                row["seed_points"][record["seed"]] += points
                row["seed_matches"][record["seed"]] += 1
    standings = [
        Standing(name, row["matches"], row["points"], row["wins"], row["losses"], row["chops"],
                 row["bb"] / row["matches"] if row["matches"] else 0.0,
                 tuple(row["seed_points"][s] for s in seeds),
                 tuple(row["seed_matches"][s] for s in seeds))
        for name, row in totals.items()
    ]
    return sorted(standings, key=lambda s: (-s.points, -s.avg_bb_per_deal, s.name))


_POOL_COMPETITORS: Sequence[tuple[str, Callable[[], object]]] = ()


def _pairing_work(task: tuple[int, int, object, int]) -> dict:
    left, right, seed, deals = task
    (left_name, left_factory), (right_name, right_factory) = (
        _POOL_COMPETITORS[left], _POOL_COMPETITORS[right])
    return match_record(left_name, right_name, left_factory(), right_factory(), seed, deals)


def round_robin_records(competitors: Sequence[tuple[str, Callable[[], object]]], *,
                        deals: int, seeds: Sequence[object],
                        workers: int = 1) -> list[dict]:
    """Every pairing on every seed. workers > 1 uses processes (Linux/macOS)."""
    global _POOL_COMPETITORS
    names = [name for name, _ in competitors]
    if len(competitors) < 2:
        raise ValueError("A round robin requires at least two competitors.")
    if len(set(names)) != len(names):
        raise ValueError("Competitor names must be unique.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Give at least one seed, without repeats.")
    tasks = [(i, j, seed, deals) for seed in seeds
             for i in range(len(competitors)) for j in range(i + 1, len(competitors))]
    _POOL_COMPETITORS = competitors
    if workers <= 1 or sys.platform == "win32":
        return [_pairing_work(task) for task in tasks]
    with mp.get_context("fork").Pool(workers) as pool:
        return list(pool.imap_unordered(_pairing_work, tasks, chunksize=4))


def run_round_robin(competitors: Sequence[tuple[str, Callable[[], object]]], *,
                    deals: int = NUM_DEALS, seeds: Sequence[object] = (42,),
                    workers: int = 1) -> list[Standing]:
    """Duplicate-scored round robin over one or more seeds, ranked by points."""
    records = round_robin_records(competitors, deals=deals, seeds=seeds, workers=workers)
    return standings_from_records(records, [name for name, _ in competitors], seeds)


__all__ = [
    "BIG_BLIND", "NUM_DEALS", "OfficialAction", "OfficialSeat", "STARTING_STACK", "Standing",
    "as_seat", "match_record", "play_half", "play_hand", "round_robin_records",
    "run_match", "run_round_robin", "standings_from_records", "to_bot_observation",
    "to_official_action",
]
