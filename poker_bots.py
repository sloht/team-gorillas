"""Short Deck five-card Omaha bots and equity estimation."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from hashlib import blake2b
from typing import Optional

from short_deck_plo import Action, DECK, Card, Observation, omaha_rank


def estimate_equity(
    observation: Observation,
    *,
    samples: int = 24,
) -> float:
    """Estimate showdown equity by sampling legal opponent hands and future board cards."""
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if observation.opponents <= 0:
        return 1.0
    return _cached_equity(
        tuple(observation.hole),
        tuple(observation.board),
        observation.opponents,
        samples,
    )


@lru_cache(maxsize=65_536)
def _cached_equity(
    hole: tuple[Card, ...],
    board: tuple[Card, ...],
    opponents: int,
    samples: int,
) -> float:
    # A card-state-derived seed gives all entrants common equity samples on paired deals.
    state = repr((hole, board, opponents, samples)).encode("ascii")
    seed = int.from_bytes(blake2b(state, digest_size=8).digest(), "big")
    random_source = random.Random(seed)
    known = set(hole) | set(board)
    remaining = [card for card in DECK if card not in known]
    missing_board = 5 - len(board)
    draw_count = missing_board + opponents * 5
    if draw_count > len(remaining):
        raise ValueError("Not enough unseen cards for the requested opponent count.")

    equity = 0.0
    for _ in range(samples):
        draw = random_source.sample(remaining, draw_count)
        final_board = (*board, *draw[:missing_board])
        hero_rank = omaha_rank(hole, final_board)
        opponent_ranks = []
        cursor = missing_board
        for _ in range(opponents):
            opponent_hole = draw[cursor:cursor + 5]
            cursor += 5
            opponent_ranks.append(omaha_rank(opponent_hole, final_board))
        best = max((hero_rank, *opponent_ranks))
        if hero_rank == best:
            equity += 1.0 / (1 + sum(rank == best for rank in opponent_ranks))
    return equity / samples


class PokerBot(ABC):
    """Bot interface accepted by the engine: implement act(observation) -> Action."""

    name = "PokerBot"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError


class ShortDeckPLOBot(PokerBot):
    """Equity-driven, position-aware baseline intended for iterative improvement."""

    name = "ShortDeckPLO"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(
            observation,
            samples=32 if observation.preflop else 24,
        )
        call_amount = observation.call_amount

        if call_amount:
            pot_odds = call_amount / (observation.pot + call_amount)
            margin = 0.035 if observation.in_position else 0.055
            if equity < pot_odds + margin:
                return Action.fold()
            if observation.can_raise and equity >= 0.72:
                return self._value_raise(observation, equity)
            return Action.call()

        value_threshold = 0.63 if observation.in_position else 0.68
        bluff_threshold = 0.42 if observation.in_position else 0.48
        if observation.can_raise and (
            equity >= value_threshold
            or (equity >= bluff_threshold and self.rng.random() < 0.12)
        ):
            return self._value_raise(observation, equity)
        return Action.check()

    @staticmethod
    def _value_raise(observation: Observation, equity: float) -> Action:
        maximum = observation.max_raise_to
        minimum = observation.min_raise_to
        if maximum <= observation.current_bet:
            return Action.call() if observation.call_amount else Action.check()
        if maximum < minimum:
            target = maximum
        else:
            fraction = min(0.82, 0.42 + max(0.0, equity - 0.65))
            target = max(minimum, observation.current_bet
                         + round((maximum - observation.current_bet) * fraction))
            target = min(target, maximum)
        return ShortDeckPLOBot._aggressive_action(observation, target)

    @staticmethod
    def _aggressive_action(observation: Observation, target: int) -> Action:
        if observation.current_bet:
            return Action.raise_to(target)
        return Action.bet_to(target)


class RandomBot(PokerBot):
    """Unpredictable reference player that chooses only legal basic actions."""

    name = "Random"

    def act(self, observation: Observation) -> Action:
        options = ["call"] if observation.call_amount else ["check"]
        if observation.call_amount:
            options.append("fold")
        if observation.can_raise and observation.max_raise_to >= observation.min_raise_to:
            options.append("raise")
        choice = self.rng.choice(options)
        if choice == "fold":
            return Action.fold()
        if choice == "call":
            return Action.call()
        if choice == "raise":
            target = self.rng.randint(observation.min_raise_to, observation.max_raise_to)
            return (Action.raise_to(target) if observation.current_bet
                    else Action.bet_to(target))
        return Action.check()


class CallingStationBot(PokerBot):
    """Passive reference player; calls whenever facing a bet and checks otherwise."""

    name = "CallingStation"

    def act(self, observation: Observation) -> Action:
        return Action.call() if observation.call_amount else Action.check()


class TightAggressiveBot(PokerBot):
    """A tighter value strategy that avoids marginal calls and raises strong hands."""

    name = "TightAggressive"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < max(0.38, required + 0.10):
                return Action.fold()
            if observation.can_raise and equity > 0.78:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        if observation.can_raise and equity > 0.70:
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class LooseAggressiveBot(PokerBot):
    """A high-pressure reference player for comparing aggression against equity."""

    name = "LooseAggressive"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=12)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < max(0.20, required - 0.02):
                return Action.fold()
            if observation.can_raise and equity > 0.55:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        if observation.can_raise and (equity > 0.48 or self.rng.random() < 0.20):
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class RockBot(PokerBot):
    """Very selective player that enters pots only with high estimated equity."""

    name = "Rock"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=12)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < max(0.52, required + 0.16):
                return Action.fold()
            return Action.call()
        if observation.can_raise and equity >= 0.77:
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class ManiacBot(PokerBot):
    """Highly aggressive benchmark that raises frequently and applies pressure."""

    name = "Maniac"

    def act(self, observation: Observation) -> Action:
        if observation.can_raise:
            return (
                Action.raise_to(observation.max_raise_to)
                if observation.current_bet
                else Action.bet_to(observation.max_raise_to)
            )
        if observation.call_amount:
            equity = estimate_equity(observation, samples=12)
            required = observation.call_amount / (observation.pot + observation.call_amount)
            return Action.call() if equity >= max(0.25, required - 0.05) else Action.fold()
        return Action.check()


class PotControlBot(PokerBot):
    """Passive equity-aware benchmark that checks and calls but rarely raises."""

    name = "PotControl"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=12)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            return Action.call() if equity >= required + 0.02 else Action.fold()
        if observation.can_raise and equity >= 0.88:
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class BluffBot(PokerBot):
    """Selective bluffing benchmark that mixes occasional pressure with value bets."""

    name = "Bluff"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=12)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.06:
                return Action.fold()
            if observation.can_raise and equity >= 0.75:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        if observation.can_raise and (
            equity >= 0.65 or self.rng.random() < (0.28 if observation.in_position else 0.16)
        ):
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class ValueFirstBot(PokerBot):
    """Value-heavy strategy that avoids marginal calls and uses larger value bets."""

    name = "ValueFirst"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.045:
                return Action.fold()
            if observation.can_raise and equity >= 0.79:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        if observation.can_raise and equity >= 0.68:
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class PotOddsBot(PokerBot):
    """Pot-odds caller that takes value raises only with a clear equity edge."""

    name = "PotOdds"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.015:
                return Action.fold()
            if observation.can_raise and equity >= 0.82:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        if observation.can_raise and equity >= 0.73:
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class PositionPressureBot(PokerBot):
    """Uses position to widen calls and increase pressure when acting last."""

    name = "PositionPressure"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        position_adjustment = 0.055 if observation.in_position else -0.035
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.035 - position_adjustment:
                return Action.fold()
            if observation.can_raise and equity >= 0.67 - position_adjustment:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        bluff_frequency = 0.24 if observation.in_position else 0.08
        if observation.can_raise and (
            equity >= 0.63 - position_adjustment
            or self.rng.random() < bluff_frequency
        ):
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class SPRShoveBot(PokerBot):
    """Uses stack-to-pot ratio to choose between pot-sized aggression and restraint."""

    name = "SPRShove"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        effective_stack = min(
            observation.stack,
            max(0, observation.max_raise_to - observation.street_contribution),
        )
        spr = effective_stack / max(1, observation.pot + observation.call_amount)
        raise_threshold = 0.69 if spr < 2 else 0.80
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + (0.035 if spr < 2 else 0.08):
                return Action.fold()
            if observation.can_raise and equity >= raise_threshold + 0.05:
                return Action.raise_to(observation.max_raise_to)
            return Action.call()
        if observation.can_raise and equity >= raise_threshold:
            return ShortDeckPLOBot._aggressive_action(
                observation, observation.max_raise_to
            )
        return Action.check()


class TrapValueBot(PokerBot):
    """Keeps medium-strength hands small and raises only with very strong equity."""

    name = "TrapValue"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.015:
                return Action.fold()
            if observation.can_raise and equity >= 0.88:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        if observation.can_raise and equity >= 0.82:
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class SmallBallBot(PokerBot):
    """Uses minimum legal raises for value and avoids building pots with marginal hands."""

    name = "SmallBall"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.055:
                return Action.fold()
            if observation.can_raise and equity >= 0.82:
                return Action.raise_to(min(observation.min_raise_to, observation.max_raise_to))
            return Action.call()
        if observation.can_raise and equity >= 0.72:
            return ShortDeckPLOBot._aggressive_action(
                observation,
                min(observation.min_raise_to, observation.max_raise_to),
            )
        return Action.check()


class BoardTextureBot(PokerBot):
    """Tightens on paired or coordinated boards and values made-hand equity."""

    name = "BoardTexture"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        ranks = [rank for rank, _ in observation.board]
        suits = [suit for _, suit in observation.board]
        paired = len(set(ranks)) < len(ranks)
        monotone = len(suits) >= 3 and len(set(suits)) == 1
        coordinated = len(ranks) >= 3 and max(ranks) - min(ranks) <= 4
        danger = 0.08 * paired + 0.10 * monotone + 0.06 * coordinated
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.035 + danger:
                return Action.fold()
            if observation.can_raise and equity >= 0.77 + danger:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        if observation.can_raise and equity >= 0.67 + danger:
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class RangePressureBot(PokerBot):
    """Applies small, position-aware bluffs while keeping calls tied to pot odds."""

    name = "RangePressure"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.025:
                return Action.fold()
            if observation.can_raise and equity >= 0.76:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        bluff_rate = 0.20 if observation.in_position else 0.07
        if observation.can_raise and (
            equity >= 0.70 or (equity >= 0.38 and self.rng.random() < bluff_rate)
        ):
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class AdaptiveCallBot(PokerBot):
    """Adjusts its calling threshold to pot odds and raises only at high equity."""

    name = "AdaptiveCall"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        if observation.call_amount:
            pot_odds = observation.call_amount / (observation.pot + observation.call_amount)
            uncertainty_margin = 0.035 if len(observation.board) < 5 else 0.015
            if equity < pot_odds + uncertainty_margin:
                return Action.fold()
            if observation.can_raise and equity >= 0.81:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        if observation.can_raise and equity >= 0.71:
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class DelayedAggressorBot(PokerBot):
    """Checks more often on the flop, then applies pressure on later streets."""

    name = "DelayedAggressor"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.04:
                return Action.fold()
            if observation.can_raise and equity >= 0.82:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        threshold = 0.78 if len(observation.board) == 3 else 0.66
        bluff_rate = 0.12 if len(observation.board) >= 4 and observation.in_position else 0.0
        if observation.can_raise and (
            equity >= threshold or self.rng.random() < bluff_rate
        ):
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class BlockerBluffBot(PokerBot):
    """Uses visible flush blockers to select occasional bluffs on suited boards."""

    name = "BlockerBluff"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=16)
        board_suits = [suit for _, suit in observation.board]
        flush_suit = None
        if len(board_suits) >= 3:
            flush_suit = max(set(board_suits), key=board_suits.count)
        blocks_flush = flush_suit is not None and sum(
            suit == flush_suit for _, suit in observation.hole
        ) >= 2
        if observation.call_amount:
            required = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < required + 0.04:
                return Action.fold()
            if observation.can_raise and equity >= 0.80:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        bluff = blocks_flush and observation.in_position and equity >= 0.32
        if observation.can_raise and (equity >= 0.70 or bluff):
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


class FieldBalancedBot(PokerBot):
    """Balanced field strategy combining pot odds, position, and measured aggression."""

    name = "FieldBalanced"

    def act(self, observation: Observation) -> Action:
        equity = estimate_equity(observation, samples=20)
        position_edge = 0.025 if observation.in_position else -0.015
        if observation.call_amount:
            pot_odds = observation.call_amount / (observation.pot + observation.call_amount)
            if equity < pot_odds + 0.04 - position_edge:
                return Action.fold()
            if observation.can_raise and equity >= 0.76 - position_edge:
                return ShortDeckPLOBot._value_raise(observation, equity)
            return Action.call()
        value_threshold = 0.68 - position_edge
        bluff = observation.in_position and 0.36 <= equity < value_threshold
        if observation.can_raise and (equity >= value_threshold or (bluff and self.rng.random() < 0.16)):
            return ShortDeckPLOBot._value_raise(observation, equity)
        return Action.check()


@dataclass(frozen=True)
class StrategyProfile:
    call_margin: float
    call_floor: float
    raise_threshold: float
    raise_when_called: float
    bluff_threshold: float
    bluff_rate: float
    bet_fraction: float
    position_edge: float
    texture_sensitivity: float
    preflop_adjustment: float
    late_street_adjustment: float
    equity_samples: int


class ParameterizedStrategyBot(PokerBot):
    """Reusable decision policy whose named subclasses use distinct profiles."""

    profile = StrategyProfile(
        call_margin=0.035,
        call_floor=0.20,
        raise_threshold=0.68,
        raise_when_called=0.79,
        bluff_threshold=0.40,
        bluff_rate=0.08,
        bet_fraction=0.65,
        position_edge=0.025,
        texture_sensitivity=0.04,
        preflop_adjustment=0.015,
        late_street_adjustment=0.0,
        equity_samples=12,
    )

    def act(self, observation: Observation) -> Action:
        profile = self.profile
        equity = estimate_equity(observation, samples=profile.equity_samples)
        texture = self._texture_danger(observation) * profile.texture_sensitivity
        position = profile.position_edge if observation.in_position else -profile.position_edge
        street = (
            profile.preflop_adjustment if observation.preflop
            else profile.late_street_adjustment if len(observation.board) >= 4
            else 0.0
        )

        if observation.call_amount:
            pot_odds = observation.call_amount / (observation.pot + observation.call_amount)
            call_threshold = max(
                profile.call_floor,
                pot_odds + profile.call_margin + texture + street - position,
            )
            if equity < call_threshold:
                return Action.fold()
            if observation.can_raise and equity >= profile.raise_when_called + texture:
                return self._raise(observation)
            return Action.call()

        raise_threshold = profile.raise_threshold + texture + street - position
        value_bet = equity >= raise_threshold
        bluff = (
            equity >= profile.bluff_threshold + texture
            and self.rng.random() < profile.bluff_rate
        )
        if observation.can_raise and (value_bet or bluff):
            return self._raise(observation)
        return Action.check()

    @staticmethod
    def _texture_danger(observation: Observation) -> float:
        ranks = [rank for rank, _ in observation.board]
        suits = [suit for _, suit in observation.board]
        danger = 0.0
        if len(set(ranks)) < len(ranks):
            danger += 0.8
        if len(suits) >= 3 and max(suits.count(suit) for suit in set(suits)) >= 3:
            danger += 1.0
        if len(ranks) >= 3 and max(ranks) - min(ranks) <= 4:
            danger += 0.6
        return danger

    def _raise(self, observation: Observation) -> Action:
        minimum = observation.min_raise_to
        maximum = observation.max_raise_to
        if maximum < minimum:
            target = maximum
        else:
            target = round(
                observation.current_bet
                + (maximum - observation.current_bet) * self.profile.bet_fraction
            )
            target = min(maximum, max(minimum, target))
        return ShortDeckPLOBot._aggressive_action(observation, target)


_PROFILE_FAMILIES = (
    ("EquityValue", 0.025, 0.20, 0.66, 0.78, 0.34, 0.04, 0.68, 0.025, 0.035, 0.015, 0.00),
    ("TightValue", 0.075, 0.40, 0.76, 0.86, 0.50, 0.015, 0.58, 0.020, 0.055, 0.045, 0.00),
    ("PotOddsCaller", 0.005, 0.16, 0.76, 0.88, 0.45, 0.025, 0.54, 0.015, 0.025, 0.005, 0.00),
    ("PositionValue", 0.030, 0.22, 0.68, 0.79, 0.34, 0.07, 0.66, 0.075, 0.035, 0.010, 0.00),
    ("BluffPressure", 0.020, 0.18, 0.63, 0.78, 0.28, 0.25, 0.73, 0.045, 0.025, 0.005, 0.00),
    ("SmallBall", 0.065, 0.30, 0.70, 0.84, 0.42, 0.025, 0.28, 0.025, 0.035, 0.025, 0.00),
    ("PotShove", 0.045, 0.28, 0.68, 0.75, 0.38, 0.055, 1.00, 0.030, 0.045, 0.015, 0.00),
    ("TextureAware", 0.035, 0.24, 0.67, 0.78, 0.40, 0.04, 0.65, 0.035, 0.12, 0.015, 0.00),
    ("LateStreet", 0.035, 0.23, 0.68, 0.79, 0.39, 0.055, 0.70, 0.035, 0.04, 0.035, -0.035),
    ("PreflopSelective", 0.045, 0.30, 0.69, 0.81, 0.40, 0.035, 0.62, 0.025, 0.035, 0.09, 0.00),
    ("DrawChaser", -0.005, 0.12, 0.73, 0.87, 0.37, 0.045, 0.60, 0.015, 0.02, -0.02, 0.025),
    ("TrapValue", 0.025, 0.22, 0.80, 0.91, 0.43, 0.015, 0.46, 0.025, 0.04, 0.01, 0.00),
    ("RangeMixer", 0.025, 0.20, 0.67, 0.80, 0.32, 0.13, 0.70, 0.035, 0.04, 0.02, 0.00),
    ("DefensiveCaller", 0.055, 0.35, 0.76, 0.88, 0.48, 0.01, 0.52, 0.015, 0.045, 0.02, 0.00),
    ("AggressiveValue", 0.010, 0.14, 0.59, 0.72, 0.26, 0.10, 0.82, 0.045, 0.02, -0.005, 0.00),
    ("BalancedField", 0.035, 0.23, 0.68, 0.79, 0.37, 0.065, 0.67, 0.04, 0.035, 0.015, 0.00),
    ("BlockerPressure", 0.030, 0.22, 0.70, 0.81, 0.36, 0.09, 0.72, 0.035, 0.06, 0.01, 0.00),
    ("UncertaintyAware", 0.045, 0.28, 0.71, 0.83, 0.41, 0.035, 0.61, 0.025, 0.035, 0.025, 0.00),
)


def _make_profile(family: tuple[object, ...], variant: int) -> StrategyProfile:
    (
        _,
        call_margin,
        call_floor,
        raise_threshold,
        raise_when_called,
        bluff_threshold,
        bluff_rate,
        bet_fraction,
        position_edge,
        texture_sensitivity,
        preflop_adjustment,
        late_street_adjustment,
    ) = family
    offset = variant - 4.5
    return StrategyProfile(
        call_margin=max(-0.02, float(call_margin) + offset * 0.004),
        call_floor=max(0.08, min(0.75, float(call_floor) + offset * 0.007)),
        raise_threshold=max(0.30, min(0.95, float(raise_threshold) + offset * 0.006)),
        raise_when_called=max(0.35, min(0.98, float(raise_when_called) + offset * 0.005)),
        bluff_threshold=max(0.15, min(0.85, float(bluff_threshold) + offset * 0.005)),
        bluff_rate=max(0.0, min(0.60, float(bluff_rate) + offset * 0.018)),
        bet_fraction=max(0.15, min(1.0, float(bet_fraction) + offset * 0.025)),
        position_edge=max(0.0, float(position_edge) + (variant % 3) * 0.008),
        texture_sensitivity=max(0.0, float(texture_sensitivity) + (variant % 4) * 0.012),
        preflop_adjustment=max(-0.05, float(preflop_adjustment) + (variant % 5) * 0.008),
        late_street_adjustment=float(late_street_adjustment) + (variant % 4 - 1.5) * 0.01,
        equity_samples=8 + (variant % 4) * 2,
    )


def _build_strategy_variants() -> tuple[type[ParameterizedStrategyBot], ...]:
    variants = []
    for family in _PROFILE_FAMILIES:
        family_name = str(family[0])
        family_variants = []
        for variant in range(1, 11):
            name = f"{family_name}{variant:02d}"
            strategy_class = type(
                name,
                (ParameterizedStrategyBot,),
                {
                    "__module__": __name__,
                    "__qualname__": name,
                    "__doc__": f"{family_name} profile, parameter variation {variant}.",
                    "name": name,
                    "profile": _make_profile(family, variant - 1),
                },
            )
            globals()[name] = strategy_class
            variants.append(strategy_class)
            family_variants.append(strategy_class)
        if len({strategy.name for strategy in family_variants}) != 10:
            raise RuntimeError(f"{family_name} generated duplicate strategy names.")
        if len({strategy.profile for strategy in family_variants}) != 10:
            raise RuntimeError(f"{family_name} generated duplicate strategy profiles.")
    return tuple(variants)


GENERATED_STRATEGY_CLASSES = _build_strategy_variants()
GENERATED_STRATEGIES = tuple(
    (strategy_class.name, strategy_class) for strategy_class in GENERATED_STRATEGY_CLASSES
)


__all__ = [
    "AdaptiveCallBot",
    "BlockerBluffBot",
    "BoardTextureBot",
    "BluffBot",
    "CallingStationBot",
    "DelayedAggressorBot",
    "FieldBalancedBot",
    "LooseAggressiveBot",
    "ManiacBot",
    "ParameterizedStrategyBot",
    "PotOddsBot",
    "PotControlBot",
    "PositionPressureBot",
    "PokerBot",
    "RandomBot",
    "StrategyProfile",
    "GENERATED_STRATEGIES",
    "GENERATED_STRATEGY_CLASSES",
    "RockBot",
    "SPRShoveBot",
    "SmallBallBot",
    "ShortDeckPLOBot",
    "TightAggressiveBot",
    "TrapValueBot",
    "ValueFirstBot",
    "estimate_equity",
] + [name for name, _ in GENERATED_STRATEGIES]
