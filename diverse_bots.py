"""Opponents that decide in structurally different ways from the equity-threshold bots.

Every other strategy in this repo estimates its chance of winning by sampling and
compares it with fold/call/raise thresholds. These bots do not sample at all:

* RuleTight / RuleLoose   - fixed rules on the made-hand category and draws
* CheckRaiser             - checks strong hands to raise when bet into (traps)
* Floater                 - line-based: calls a flop bet in position, bets the turn
                            when checked to, gives up when bet into again
* DelayedProbe            - never bets the flop; bets the turn and river when checked to
* MixedStrategy / MixedLoose - randomises between actions on purpose, by hand-strength bucket
* NitFolder               - folds to almost every bet, bets only strong hands
* PreflopJammer           - raises preflop to the maximum with decent hands, else folds
* Exploiter / ExploiterAggro - models the opponent across hands (fold-to-bet,
                            aggression, showdown bluffs) and adjusts to exploit it

They play official observations directly (official_match.as_seat passes them
through) and are fast: no equity sampling, only a few table lookups.
"""

from __future__ import annotations

import random
from itertools import combinations

import official_rules  # noqa: F401  (puts ./official on sys.path)
from evaluator import (FLUSH, FULL_HOUSE, ONE_PAIR, STRAIGHT, THREE_OF_A_KIND, TWO_PAIR,
                       category_of, get_table, omaha_score)
from evolved_policy import showdown_percentile
from game import Action, Street

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


# --------------------------------------------------------------------------- #
# Cheap hand reading (no sampling)
# --------------------------------------------------------------------------- #

def preflop_strength(hole) -> float:
    """Roughly 0..1: the best two-card combinations, with penalties for wasted cards."""
    combos = []
    for a, b in combinations(hole, 2):
        ra, rb = a >> 2, b >> 2
        if ra == rb:
            score = 0.55 + 0.05 * ra
        else:
            score = 0.10 + 0.025 * (ra + rb)
            if (a & 3) == (b & 3):
                score += 0.12
            if abs(ra - rb) <= 2:
                score += 0.06
        combos.append(score)
    combos.sort(reverse=True)
    strength = 0.65 * combos[0] + 0.35 * combos[1]
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    if max(suits.count(s) for s in set(suits)) >= 4:
        strength -= 0.10  # the flush cards block each other
    if max(ranks.count(r) for r in set(ranks)) >= 3:
        strength -= 0.10  # only two of them can play
    return max(0.0, min(1.0, strength))


def made_score(hole, board) -> int:
    """Best official score with exactly 2 hole and 3 board cards (board of 3 to 5)."""
    if len(board) == 5:
        return omaha_score(hole, board)
    table = get_table()
    hole_bits = [1 << c for c in hole]
    board_bits = [1 << c for c in board]
    return max(table[hole_bits[i] | hole_bits[j] | sum(board_bits[k] for k in triple)]
               for i, j in combinations(range(5), 2)
               for triple in combinations(range(len(board)), 3))


def flush_draw(hole, board) -> bool:
    if len(board) >= 5:
        return False
    for suit in range(4):
        if sum((c & 3) == suit for c in hole) >= 2 and sum((c & 3) == suit for c in board) == 2:
            return True
    return False


CATEGORY_STRENGTH = {0: 0.10, ONE_PAIR: 0.35, TWO_PAIR: 0.55, STRAIGHT: 0.65,
                     THREE_OF_A_KIND: 0.75, FULL_HOUSE: 0.90, FLUSH: 0.85, 7: 0.97, 8: 1.0}


def strength(obs) -> float:
    if not obs.board:
        return preflop_strength(obs.hole_cards)
    s = CATEGORY_STRENGTH[category_of(made_score(obs.hole_cards, obs.board))]
    if flush_draw(obs.hole_cards, obs.board):
        s = max(s, 0.45)
    return s


def category(obs) -> int:
    return category_of(made_score(obs.hole_cards, obs.board)) if obs.board else -1


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #

class DiverseBot:
    plays_official = True
    name = "DiverseBot"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        self.rng = random.Random(seed)

    @staticmethod
    def fold(obs) -> Action:
        return Action.FOLD if Action.FOLD in obs.legal_actions else Action.CHECK_CALL

    @staticmethod
    def raise_(obs, big: bool = True) -> Action:
        wanted = Action.RAISE_POT if big else Action.RAISE_HALF
        return wanted if wanted in obs.legal_actions else Action.CHECK_CALL

    @staticmethod
    def pot_odds(obs) -> float:
        return obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0

    def act(self, obs) -> Action:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# The bots
# --------------------------------------------------------------------------- #

class RuleBot(DiverseBot):
    """Fixed rules on the made-hand category; `loose` widens every range."""

    def __init__(self, seed: int | None = None, loose: bool = False) -> None:
        super().__init__(seed)
        self.loose = loose
        self.name = "RuleLoose" if loose else "RuleTight"

    def act(self, obs) -> Action:
        shift = -0.12 if self.loose else 0.0
        if not obs.board:
            s = preflop_strength(obs.hole_cards)
            if s >= 0.72 + shift:
                return self.raise_(obs)
            if s >= 0.50 + shift or not obs.to_call:
                return Action.CHECK_CALL
            return self.fold(obs)
        cat = category(obs)
        if cat >= THREE_OF_A_KIND:
            return self.raise_(obs)
        if cat >= TWO_PAIR:
            return Action.CHECK_CALL if obs.to_call else self.raise_(obs, big=False)
        if flush_draw(obs.hole_cards, obs.board) and self.pot_odds(obs) <= 0.30 + shift * -1:
            return Action.CHECK_CALL
        if cat == ONE_PAIR and obs.to_call <= obs.pot * (0.5 if self.loose else 0.3):
            return Action.CHECK_CALL
        return self.fold(obs) if obs.to_call else Action.CHECK_CALL


class CheckRaiser(DiverseBot):
    """Checks strong hands after the flop, then raises when the opponent bets."""

    name = "CheckRaiser"

    def act(self, obs) -> Action:
        if not obs.board:
            s = preflop_strength(obs.hole_cards)
            if s >= 0.80:
                return self.raise_(obs, big=False)
            return Action.CHECK_CALL if s >= 0.35 or not obs.to_call else self.fold(obs)
        cat = category(obs)
        if cat >= STRAIGHT:
            if obs.to_call:
                return self.raise_(obs)
            return Action.CHECK_CALL if obs.street != Street.RIVER else self.raise_(obs, big=False)
        if cat == TWO_PAIR or flush_draw(obs.hole_cards, obs.board):
            return Action.CHECK_CALL if self.pot_odds(obs) <= 0.35 else self.fold(obs)
        if cat == ONE_PAIR and obs.to_call <= obs.pot * 0.25:
            return Action.CHECK_CALL
        return self.fold(obs) if obs.to_call else Action.CHECK_CALL


class Floater(DiverseBot):
    """Calls a flop bet in position with a playable hand, bets the turn if checked to,
    and gives up when bet into again without two pair or better."""

    name = "Floater"

    def act(self, obs) -> Action:
        if not obs.board:
            return Action.CHECK_CALL if preflop_strength(obs.hole_cards) >= 0.30 or not obs.to_call \
                else self.fold(obs)
        cat = category(obs)
        if obs.street == Street.FLOP:
            if obs.to_call:
                return Action.CHECK_CALL if (obs.is_button or cat >= ONE_PAIR) and \
                    self.pot_odds(obs) <= 0.40 else self.fold(obs)
            return Action.CHECK_CALL
        if not obs.to_call:
            return self.raise_(obs)  # bet the turn and river when checked to
        return Action.CHECK_CALL if cat >= TWO_PAIR else self.fold(obs)


class DelayedProbe(DiverseBot):
    """Never bets the flop. Bets the turn and river when checked to, with a pair or
    better or as a bluff a third of the time."""

    name = "DelayedProbe"

    def act(self, obs) -> Action:
        if not obs.board:
            s = preflop_strength(obs.hole_cards)
            if s >= 0.75:
                return self.raise_(obs, big=False)
            return Action.CHECK_CALL if s >= 0.40 or not obs.to_call else self.fold(obs)
        cat = category(obs)
        if obs.to_call:
            if cat >= TWO_PAIR:
                return Action.CHECK_CALL
            return Action.CHECK_CALL if cat == ONE_PAIR and self.pot_odds(obs) <= 0.25 else self.fold(obs)
        if obs.street == Street.FLOP:
            return Action.CHECK_CALL
        if cat >= ONE_PAIR or self.rng.random() < 0.33:
            return self.raise_(obs, big=cat < TWO_PAIR)
        return Action.CHECK_CALL


class MixedStrategy(DiverseBot):
    """Randomises between actions on purpose, with probabilities set by hand strength."""

    def __init__(self, seed: int | None = None, loose: bool = False) -> None:
        super().__init__(seed)
        self.loose = loose
        self.name = "MixedLoose" if loose else "MixedStrategy"

    def act(self, obs) -> Action:
        s = strength(obs) + (0.08 if self.loose else 0.0)
        roll = self.rng.random()
        if obs.to_call:
            fold_p = max(0.0, 0.85 - 1.6 * s + self.pot_odds(obs))
            raise_p = max(0.0, 1.4 * s - 0.75)
            if roll < fold_p:
                return self.fold(obs)
            if roll < fold_p + raise_p:
                return self.raise_(obs, big=self.rng.random() < 0.5)
            return Action.CHECK_CALL
        bet_p = min(0.9, max(0.12, 1.5 * s - 0.45))
        if roll < bet_p:
            return self.raise_(obs, big=self.rng.random() < s)
        return Action.CHECK_CALL


class NitFolder(DiverseBot):
    """Folds to almost every bet; bets only strong hands."""

    name = "NitFolder"

    def act(self, obs) -> Action:
        s = strength(obs)
        if obs.to_call:
            if s >= 0.75:
                return self.raise_(obs)
            if s >= 0.60 and self.pot_odds(obs) <= 0.30:
                return Action.CHECK_CALL
            return self.fold(obs)
        return self.raise_(obs, big=False) if s >= 0.65 else Action.CHECK_CALL


class PreflopJammer(DiverseBot):
    """Raises preflop to the maximum with decent hands, folds the rest, then calls down."""

    name = "PreflopJammer"

    def act(self, obs) -> Action:
        if not obs.board:
            if preflop_strength(obs.hole_cards) >= 0.55:
                return self.raise_(obs)
            return Action.CHECK_CALL if not obs.to_call else self.fold(obs)
        return Action.CHECK_CALL


class Exploiter(DiverseBot):
    """Models the opponent across the half and exploits it:
    bluffs more against frequent folders, value-bets thinner against stations,
    calls wider against aggressive or bluffing opponents, folds more to passive ones."""

    def __init__(self, seed: int | None = None, aggro: bool = False) -> None:
        self.aggro = aggro
        self.name = "ExploiterAggro" if aggro else "Exploiter"
        super().__init__(seed)
        self.reset()

    def set_seed(self, seed: int) -> None:
        super().set_seed(seed)
        self.reset()  # a new half: forget the old opponent

    def reset(self) -> None:
        self.opp_actions = self.opp_raises = 0
        self.bet_hands = self.opp_folds = 0
        self.sd_raises = self.sd_weak = 0
        self.hand = None
        self.seen = 0
        self.hole = ()
        self.we_raised = self.opp_raised = False

    def _observe(self, obs) -> None:
        if obs.hand_index != self.hand:
            self.hand, self.seen = obs.hand_index, 0
            self.we_raised = self.opp_raised = False
        self.hole = obs.hole_cards
        for record in obs.action_history[self.seen:]:
            if not record.is_me:
                self.opp_actions += 1
                if record.action in RAISES:
                    self.opp_raises += 1
                    self.opp_raised = True
        self.seen = len(obs.action_history)

    def on_hand_end(self, result) -> None:
        if result.hand_index != self.hand:
            return
        if self.we_raised:
            self.bet_hands += 1
            self.opp_folds += (not result.went_to_showdown and result.my_net > 0)
        if result.went_to_showdown and self.opp_raised and result.opp_hole and len(result.board) == 5:
            self.sd_raises += 1
            # below-median hand for this board (one pair or worse almost never happens here)
            self.sd_weak += showdown_percentile(result.opp_hole, result.board, self.hole) < 0.5

    def act(self, obs) -> Action:
        self._observe(obs)
        action = self._decide(obs)
        if action in RAISES:
            self.we_raised = True
        return action

    def _decide(self, obs) -> Action:
        s = strength(obs)
        fold_rate = (self.opp_folds + 4) / (self.bet_hands + 10)        # prior 0.40
        aggression = (self.opp_raises + 5) / (self.opp_actions + 20)    # prior 0.25
        bluffing = (self.sd_weak + 2.0) / (self.sd_raises + 8)          # prior 0.25
        if obs.to_call:
            need = (self.pot_odds(obs) + 0.15 - 0.6 * (bluffing - 0.25)
                    - 0.5 * (aggression - 0.25) + (0.10 if aggression < 0.15 else 0.0))
            if s >= 0.80:
                return self.raise_(obs)
            return Action.CHECK_CALL if s >= need else self.fold(obs)
        value = 0.60 + 0.5 * (fold_rate - 0.40) - (0.12 if self.aggro else 0.0)  # thinner vs stations
        if s >= value:
            return self.raise_(obs, big=s >= 0.75 or fold_rate < 0.3 or self.aggro)
        bluff = max(0.0, min(0.7, 1.6 * (fold_rate - 0.25))) + (0.30 if self.aggro else 0.0)
        if obs.board and self.rng.random() < bluff:
            return self.raise_(obs, big=fold_rate > 0.5)
        return Action.CHECK_CALL


DIVERSE_BOTS = {
    "RuleTight": lambda: RuleBot(),
    "RuleLoose": lambda: RuleBot(loose=True),
    "CheckRaiser": CheckRaiser,
    "Floater": Floater,
    "DelayedProbe": DelayedProbe,
    "MixedStrategy": lambda: MixedStrategy(),
    "MixedLoose": lambda: MixedStrategy(loose=True),
    "NitFolder": NitFolder,
    "PreflopJammer": PreflopJammer,
    "Exploiter": lambda: Exploiter(),
    "ExploiterAggro": lambda: Exploiter(aggro=True),
}

__all__ = ["DIVERSE_BOTS", "preflop_strength", "made_score", "strength"]
