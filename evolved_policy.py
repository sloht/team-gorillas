"""A tunable strategy family for the evolution search, written to drop into agent.py.

Everything between the POLICY START and POLICY END markers uses only the
official bundle's `evaluator` and `game` modules, so build_agent.py can copy
it verbatim into a tournament agent.py. With the POTSHOVE08 genome every
decision is exactly the decision of poker_bots.PotShove08 played through
official_match.OfficialSeat (tests/test_evolved_policy.py checks this).

What a genome controls, beyond PotShove08's profile:
  * separate preflop / postflop call floors, call margins, open-raise and
    re-raise thresholds, and bluff rates;
  * raise size: the pot raise when equity >= size_pot_eq, otherwise half pot
    (bluffs use bluff_pot);
  * reraise_step: extra equity needed for each raise already made this street;
  * commit_spr / commit_eq: when the stack left is small next to the pot, call
    with any equity of at least commit_eq;
  * trap_eq / trap_rate: sometimes check very strong hands before the river;
  * opponent model: the share of the opponent's actions that are raises this
    half (smoothed), which lowers the call threshold (k_call) and shifts the
    value-bet threshold (k_value) against aggressive opponents;
  * adaptation from finished hands (on_hand_end): how often the opponent folds
    to our raises (a_bluff, a_thin) and how often its showdown raises were
    below-median hands for the board (a_catch).
"""

from __future__ import annotations

import official_rules  # noqa: F401  (puts ./official on sys.path in this repo)

# ===== POLICY START (copied into agent.py by build_agent.py) =====
import random
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from hashlib import blake2b

from evaluator import SUIT_CHARS, omaha_score
from game import Action, Street


@dataclass(frozen=True)
class Genome:
    samples: int = 14
    pre_call_floor: float = 0.29750000000000004
    post_call_floor: float = 0.29750000000000004
    pre_call_margin: float = 0.055
    post_call_margin: float = 0.055
    pre_adj: float = 0.031
    late_adj: float = 0.015
    pre_raise: float = 0.6950000000000001
    post_raise: float = 0.6950000000000001
    pre_reraise: float = 0.7625
    post_reraise: float = 0.7625
    reraise_step: float = 0.0
    bluff_eq: float = 0.3925
    bluff_rate_pre: float = 0.1
    bluff_rate_post: float = 0.1
    bluff_pot: int = 1
    size_pot_eq: float = 0.0
    position_edge: float = 0.038
    texture_sens: float = 0.081
    commit_spr: float = 0.0
    commit_eq: float = 0.3
    k_call: float = 0.0
    k_value: float = 0.0
    trap_eq: float = 0.95
    trap_rate: float = 0.0
    # Adaptation (0 = off). Estimates are smoothed toward a prior and adjustments capped.
    a_bluff: float = 0.0   # bluff more vs opponents who often fold to our bets
    a_thin: float = 0.0    # value-bet thinner vs opponents who rarely fold
    a_catch: float = 0.0   # call wider vs opponents caught raising weak hands at showdown

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


POTSHOVE08 = Genome()  # every default above is PotShove08's profile

_TEAM_SUITS = "shdc"
_TEAM_DECK = tuple((rank, suit) for suit in range(4) for rank in range(6, 15))
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
_AGGR_PRIOR, _AGGR_WEIGHT = 0.25, 20.0
_FOLD_PRIOR, _FOLD_WEIGHT = 0.40, 15.0     # opponent folds to our bets
_BLUFF_PRIOR, _BLUFF_WEIGHT = 0.25, 10.0   # opponent's showdown raises made with a below-median hand
_ADJUST_CAP = 0.10
_PERCENTILE_SAMPLES = 20


def _cap(value: float) -> float:
    return max(-_ADJUST_CAP, min(_ADJUST_CAP, value))


def showdown_percentile(hole, board, known=()) -> float:
    """How strong a showdown hand was for this board: the share of random five-card
    holdings (from the unseen cards) it beats, ties counting half. Below 0.5 means a
    weaker-than-typical hand. (Absolute categories do not work here: with exactly
    two hole cards, random hands are one pair or worse only about 1% of the time.)"""
    seen = set(hole) | set(board) | set(known)
    unseen = [c for c in range(36) if c not in seen]
    state = repr((tuple(sorted(hole)), tuple(board))).encode("ascii")
    rng = random.Random(int.from_bytes(blake2b(state, digest_size=8).digest(), "big"))
    score = omaha_score(hole, board)
    total = 0.0
    for _ in range(_PERCENTILE_SAMPLES):
        other = omaha_score(rng.sample(unseen, 5), board)
        total += 1.0 if score > other else 0.5 if score == other else 0.0
    return total / _PERCENTILE_SAMPLES


def _to_team(card: int) -> tuple:
    return (card >> 2) + 6, _TEAM_SUITS.index(SUIT_CHARS[card & 3])


def _to_official(card: tuple) -> int:
    rank, suit = card
    return ((rank - 6) << 2) | SUIT_CHARS.index(_TEAM_SUITS[suit])


@lru_cache(maxsize=65_536)
def _equity(hole: tuple, board: tuple, opponents: int, samples: int) -> float:
    """Share of sampled showdowns won against one random hand (ties split).

    Samples in the team repo's card order with a card-derived seed, so the
    numbers match poker_bots.estimate_equity exactly.
    """
    state = repr((hole, board, opponents, samples)).encode("ascii")
    rng = random.Random(int.from_bytes(blake2b(state, digest_size=8).digest(), "big"))
    known = set(hole) | set(board)
    remaining = [card for card in _TEAM_DECK if card not in known]
    missing_board = 5 - len(board)
    draw_count = missing_board + opponents * 5
    official_hole = tuple(_to_official(card) for card in hole)
    official_board = tuple(_to_official(card) for card in board)
    official_remaining = {card: _to_official(card) for card in remaining}
    equity = 0.0
    for _ in range(samples):
        draw = [official_remaining[card] for card in rng.sample(remaining, draw_count)]
        final_board = (*official_board, *draw[:missing_board])
        hero = omaha_score(official_hole, final_board)
        opponent_ranks = []
        cursor = missing_board
        for _ in range(opponents):
            opponent_ranks.append(omaha_score(draw[cursor:cursor + 5], final_board))
            cursor += 5
        best = max((hero, *opponent_ranks))
        if hero == best:
            equity += 1.0 / (1 + sum(rank == best for rank in opponent_ranks))
    return equity / samples


def _texture_danger(board: tuple) -> float:
    ranks = [rank for rank, _ in board]
    suits = [suit for _, suit in board]
    danger = 0.0
    if len(set(ranks)) < len(ranks):
        danger += 0.8
    if len(suits) >= 3 and max(suits.count(suit) for suit in set(suits)) >= 3:
        danger += 1.0
    if len(ranks) >= 3 and max(ranks) - min(ranks) <= 4:
        danger += 0.6
    return danger


class EvolvedPolicy:
    """Plays official observations directly (official_match.as_seat passes it through).

    Keeps its random source and opponent model for one half; set_seed (called
    by the match runner at the start of every half) and new_half reset both.
    """

    plays_official = True

    def __init__(self, genome: Genome = POTSHOVE08, name: str = "Evolved",
                 seed: int | None = None) -> None:
        self.genome = genome
        self.name = name
        self.new_half(seed)

    def new_half(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.opp_actions = 0
        self.opp_raises = 0
        self.bet_hands = 0        # finished hands in which we raised
        self.opp_folds = 0        # ... that the opponent then folded
        self.sd_raises = 0        # showdowns where the opponent had raised
        self.sd_bluffs = 0        # ... holding one pair or worse
        self._hand = None
        self._seen = 0
        self._we_raised = False
        self._opp_raised = False
        self._hole = ()

    def set_seed(self, seed: int) -> None:
        self.new_half(seed)

    def _observe(self, obs) -> None:
        if obs.hand_index != self._hand:
            self._hand, self._seen = obs.hand_index, 0
            self._we_raised = self._opp_raised = False
        self._hole = obs.hole_cards
        history = obs.action_history
        for record in history[self._seen:]:
            if not record.is_me:
                self.opp_actions += 1
                if record.action in _RAISES:
                    self.opp_raises += 1
                    self._opp_raised = True
        self._seen = len(history)

    def on_hand_end(self, result) -> None:
        """Learn from a finished hand (the tournament calls this; so does the match runner)."""
        if result.hand_index != self._hand:
            return  # we never acted in that hand
        if self._we_raised:
            self.bet_hands += 1
            if not result.went_to_showdown and result.my_net > 0:
                self.opp_folds += 1
        if result.went_to_showdown and self._opp_raised and result.opp_hole \
                and len(result.board) == 5:
            self.sd_raises += 1
            if showdown_percentile(result.opp_hole, result.board, self._hole) < 0.5:
                self.sd_bluffs += 1
        self._we_raised = self._opp_raised = False

    def opponent_aggression(self) -> float:
        return ((self.opp_raises + _AGGR_PRIOR * _AGGR_WEIGHT)
                / (self.opp_actions + _AGGR_WEIGHT))

    def opponent_fold_rate(self) -> float:
        return (self.opp_folds + _FOLD_PRIOR * _FOLD_WEIGHT) / (self.bet_hands + _FOLD_WEIGHT)

    def opponent_bluff_rate(self) -> float:
        return (self.sd_bluffs + _BLUFF_PRIOR * _BLUFF_WEIGHT) / (self.sd_raises + _BLUFF_WEIGHT)

    def _raise(self, obs, equity: float, bluffing: bool) -> Action:
        half, pot = obs.raise_to[Action.RAISE_HALF], obs.raise_to[Action.RAISE_POT]
        if half == pot:
            return Action.RAISE_HALF  # the same total either way
        big = bool(self.genome.bluff_pot) if bluffing else equity >= self.genome.size_pot_eq
        return Action.RAISE_POT if big else Action.RAISE_HALF

    def act(self, obs, samples: int | None = None) -> Action:
        action = self._decide(obs, samples)
        if action in _RAISES:
            self._we_raised = True
        return action

    def _decide(self, obs, samples: int | None) -> Action:
        g = self.genome
        self._observe(obs)
        fold_dev = self.opponent_fold_rate() - _FOLD_PRIOR
        catch = _cap(g.a_catch * (self.opponent_bluff_rate() - _BLUFF_PRIOR))
        thin = _cap(g.a_thin * fold_dev)
        hole = tuple(_to_team(card) for card in obs.hole_cards)
        board = tuple(_to_team(card) for card in obs.board)
        preflop = not board
        can_raise = Action.RAISE_HALF in obs.legal_actions and Action.RAISE_POT in obs.legal_actions
        equity = _equity(hole, board, 1, samples or g.samples)
        texture = _texture_danger(board) * g.texture_sens
        position = g.position_edge if obs.is_button else -g.position_edge
        street = g.pre_adj if preflop else g.late_adj if len(board) >= 4 else 0.0
        aggression = self.opponent_aggression() - _AGGR_PRIOR

        if obs.to_call:
            pot_odds = obs.to_call / (obs.pot + obs.to_call)
            floor = g.pre_call_floor if preflop else g.post_call_floor
            margin = g.pre_call_margin if preflop else g.post_call_margin
            threshold = max(floor, pot_odds + margin + texture + street - position
                            - g.k_call * aggression - catch)
            if g.commit_spr > 0 and obs.my_stack <= g.commit_spr * obs.pot:
                threshold = min(threshold, g.commit_eq)
            if equity < threshold:
                return Action.FOLD if Action.FOLD in obs.legal_actions else Action.CHECK_CALL
            reraise = ((g.pre_reraise if preflop else g.post_reraise) + texture
                       + g.reraise_step * obs.raises_this_street)
            if can_raise and equity >= reraise:
                return self._raise(obs, equity, False)
            return Action.CHECK_CALL

        value_bet = equity >= ((g.pre_raise if preflop else g.post_raise) + texture + street
                               - position + g.k_value * aggression + thin)
        bluff_rate = g.bluff_rate_pre if preflop else g.bluff_rate_post
        if g.a_bluff:
            bluff_rate = min(0.6, max(0.0, bluff_rate + g.a_bluff * fold_dev))
        bluff = equity >= g.bluff_eq + texture and self.rng.random() < bluff_rate
        if (value_bet and g.trap_rate > 0 and obs.street != Street.RIVER
                and equity >= g.trap_eq and self.rng.random() < g.trap_rate):
            return Action.CHECK_CALL  # slow-play a very strong hand
        if can_raise and (value_bet or bluff):
            return self._raise(obs, equity, not value_bet)
        return Action.CHECK_CALL
# ===== POLICY END =====


# Search space for the evolution search: (low, high) for every gene. The first
# ranges are kept so generations searched under them can be rebuilt exactly; the
# wide ones loosen the limits that champions of both lineages ran into.
NARROW_RANGES: dict[str, tuple[float, float]] = {
    "samples": (8, 16),
    "pre_call_floor": (0.10, 0.50),
    "post_call_floor": (0.10, 0.50),
    "pre_call_margin": (-0.03, 0.12),
    "post_call_margin": (-0.03, 0.12),
    "pre_adj": (-0.05, 0.10),
    "late_adj": (-0.05, 0.08),
    "pre_raise": (0.50, 0.85),
    "post_raise": (0.50, 0.85),
    "pre_reraise": (0.60, 0.95),
    "post_reraise": (0.60, 0.95),
    "reraise_step": (0.0, 0.08),
    "bluff_eq": (0.15, 0.55),
    "bluff_rate_pre": (0.0, 0.30),
    "bluff_rate_post": (0.0, 0.35),
    "bluff_pot": (0, 1),
    "size_pot_eq": (0.0, 0.95),
    "position_edge": (0.0, 0.08),
    "texture_sens": (0.0, 0.15),
    "commit_spr": (0.0, 1.5),
    "commit_eq": (0.20, 0.45),
    "k_call": (0.0, 0.6),
    "k_value": (-0.3, 0.3),
    "trap_eq": (0.75, 0.98),
    "trap_rate": (0.0, 0.40),
    "a_bluff": (0.0, 0.6),
    "a_thin": (0.0, 0.3),
    "a_catch": (0.0, 0.5),
}
WIDE_RANGES: dict[str, tuple[float, float]] = {
    **NARROW_RANGES,
    "pre_call_floor": (0.05, 0.50),
    "pre_call_margin": (-0.08, 0.12),
    "post_call_margin": (-0.03, 0.25),
    "pre_adj": (-0.10, 0.10),
    "late_adj": (-0.05, 0.15),
    "pre_raise": (0.40, 0.85),
    "post_raise": (0.40, 0.85),
    "bluff_eq": (0.0, 0.55),
    "bluff_rate_pre": (0.0, 0.45),
    "bluff_rate_post": (0.0, 0.50),
    "a_catch": (0.0, 0.8),
}
GENE_RANGES: dict[str, tuple[float, float]] = dict(WIDE_RANGES)
INT_GENES = {"samples", "bluff_pot"}


def use_gene_ranges(wide: bool) -> None:
    """Switch the search space in place (every module sees the same dict)."""
    GENE_RANGES.clear()
    GENE_RANGES.update(WIDE_RANGES if wide else NARROW_RANGES)


def _clip(name: str, value: float):
    low, high = GENE_RANGES[name]
    value = min(high, max(low, value))
    if name == "samples":
        return int(round(value / 2) * 2)  # even sample counts only
    if name in INT_GENES:
        return int(round(value))
    return round(value, 4)


def random_genome(rng: random.Random) -> Genome:
    return Genome(**{name: _clip(name, rng.uniform(low, high))
                     for name, (low, high) in GENE_RANGES.items()})


def mutate(genome: Genome, rng: random.Random, rate: float = 0.3, scale: float = 0.12) -> Genome:
    """Nudge about `rate` of the genes by up to `scale` of their range (Gaussian)."""
    changes = {}
    for name, (low, high) in GENE_RANGES.items():
        if rng.random() < rate:
            if name == "bluff_pot":
                changes[name] = 1 - getattr(genome, name)
            else:
                changes[name] = _clip(name, getattr(genome, name) + rng.gauss(0, scale) * (high - low))
    if not changes:  # always change something
        name = rng.choice([n for n in GENE_RANGES if n != "bluff_pot"])
        low, high = GENE_RANGES[name]
        changes[name] = _clip(name, getattr(genome, name) + rng.gauss(0, scale) * (high - low))
    return replace(genome, **changes)


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    return Genome(**{name: getattr(a if rng.random() < 0.5 else b, name) for name in GENE_RANGES})


__all__ = ["EvolvedPolicy", "GENE_RANGES", "Genome", "POTSHOVE08", "crossover", "mutate",
           "use_gene_ranges",
           "random_genome"]
