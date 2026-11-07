"""Short Deck five-card Pot-Limit Omaha engine and paired match runner."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional, Sequence

Card = tuple[int, int]  # (rank, suit); ranks are 6..14, suits are 0..3
DECK: tuple[Card, ...] = tuple((rank, suit) for suit in range(4) for rank in range(6, 15))


def rank_five(cards: Sequence[Card]) -> tuple[int, ...]:
    """Return a comparable Short Deck rank; flushes beat full houses."""
    if len(cards) != 5:
        raise ValueError("A five-card hand is required.")

    ranks = [card[0] for card in cards]
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({card[1] for card in cards}) == 1
    unique = sorted(counts, reverse=True)
    straight_high = 0
    if len(unique) == 5:
        if unique[0] - unique[-1] == 4:
            straight_high = unique[0]
        elif unique == [14, 9, 8, 7, 6]:
            straight_high = 9

    if flush and straight_high:
        return (8, straight_high)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if flush:
        return (6, *sorted(ranks, reverse=True))
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (5, groups[0][1], groups[1][1])
    if straight_high:
        return (4, straight_high)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    if groups[0][0] == 2 and groups[1][0] == 2:
        pairs = sorted((groups[0][1], groups[1][1]), reverse=True)
        kicker = next(r for r in ranks if r not in pairs)
        return (2, *pairs, kicker)
    if groups[0][0] == 2:
        pair = groups[0][1]
        return (1, pair, *sorted((r for r in ranks if r != pair), reverse=True))
    return (0, *sorted(ranks, reverse=True))


def omaha_rank(hole: Sequence[Card], board: Sequence[Card]) -> tuple[int, ...]:
    """Rank a five-card Omaha hand using exactly two hole and three board cards."""
    if len(hole) != 5:
        raise ValueError("Five-card Omaha requires exactly five hole cards.")
    if not 3 <= len(board) <= 5:
        raise ValueError("The board must contain between three and five cards.")
    return max(
        rank_five((*hole_pair, *board_triple))
        for hole_pair in combinations(hole, 2)
        for board_triple in combinations(board, 3)
    )


@dataclass(frozen=True)
class Action:
    """For bet/raise, amount is the target street contribution, not the increment."""

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
        return self.seat == self.button


@dataclass
class _Player:
    name: str
    bot: object
    stack: int
    hole: tuple[Card, ...] = ()
    street: int = 0
    total: int = 0
    folded: bool = False
    all_in: bool = False


@dataclass(frozen=True)
class MatchResult:
    bot_names: tuple[str, str]
    hands_played: int
    net_chips: tuple[int, int]
    hand_wins: tuple[int, int]

    @property
    def winner(self) -> Optional[str]:
        if self.net_chips[0] == self.net_chips[1]:
            return None
        return self.bot_names[0] if self.net_chips[0] > self.net_chips[1] else self.bot_names[1]


@dataclass(frozen=True)
class Standing:
    name: str
    matches: int
    match_wins: int
    match_losses: int
    match_ties: int
    net_chips: int
    hand_wins: int
    seed_match_wins: tuple[int, ...] = ()
    seed_match_counts: tuple[int, ...] = ()

    @property
    def seed_win_rate_range(self) -> Optional[tuple[float, float]]:
        if not self.seed_match_counts:
            return None
        rates = tuple(
            wins / matches
            for wins, matches in zip(self.seed_match_wins, self.seed_match_counts)
            if matches
        )
        return (min(rates), max(rates)) if rates else None


class _Hand:
    def __init__(
        self,
        players: list[_Player],
        button: int,
        small_blind: int,
        big_blind: int,
        deck: Sequence[Card],
    ) -> None:
        self.players = players
        self.button = button
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = list(deck)
        self.board: list[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.last_raise_size = big_blind

    def play(self) -> None:
        for player in self.players:
            player.street = player.total = 0
            player.folded = player.all_in = False
            player.hole = ()

        for _ in range(5):
            for seat in (self.button, 1 - self.button):
                self.players[seat].hole += (self.deck.pop(),)

        small_seat = self.button
        big_seat = 1 - self.button
        self._post(small_seat, self.small_blind)
        self._post(big_seat, self.big_blind)
        self.current_bet = max(player.street for player in self.players)
        self._betting_round(self._clockwise(big_seat))

        if self._contender_count() > 1:
            for count in (3, 1, 1):
                self.board.extend(self.deck.pop() for _ in range(count))
                for player in self.players:
                    player.street = 0
                self.current_bet = 0
                self.last_raise_size = self.big_blind
                if self._actionable_count() > 1:
                    self._betting_round((self.button, 1 - self.button))
        self._award()

    def _clockwise(self, after: int) -> list[int]:
        return [(after + offset) % 2 for offset in range(1, 3)]

    def _contenders(self) -> list[int]:
        return [i for i, player in enumerate(self.players) if not player.folded]

    def _contender_count(self) -> int:
        return len(self._contenders())

    def _actionable_count(self) -> int:
        return sum(not player.folded and not player.all_in for player in self.players)

    def _post(self, seat: int, amount: int) -> None:
        self._contribute(seat, min(amount, self.players[seat].stack))

    def _contribute(self, seat: int, amount: int) -> None:
        player = self.players[seat]
        player.stack -= amount
        player.street += amount
        player.total += amount
        self.pot += amount
        player.all_in = player.stack == 0

    def _observation(self, seat: int, can_raise: bool) -> Observation:
        player = self.players[seat]
        call_amount = max(0, self.current_bet - player.street)
        pot_limit_add = call_amount * 2 + self.pot
        effective_opponent_total = max(
            (other.street + other.stack for i, other in enumerate(self.players)
             if i != seat and not other.folded and not other.all_in),
            default=player.street + player.stack,
        )
        max_raise_to = min(
            player.street + player.stack,
            player.street + pot_limit_add,
            effective_opponent_total,
        )
        min_raise_to = self.current_bet + self.last_raise_size if self.current_bet else self.big_blind
        return Observation(
            seat=seat,
            button=self.button,
            hole=player.hole,
            board=tuple(self.board),
            pot=self.pot,
            stack=player.stack,
            street_contribution=player.street,
            current_bet=self.current_bet,
            call_amount=call_amount,
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
            can_raise=can_raise and max_raise_to > self.current_bet,
            big_blind=self.big_blind,
            opponents=self._contender_count() - 1,
        )

    def _betting_round(self, order: Sequence[int]) -> None:
        pending = [i for i in order if not self.players[i].folded and not self.players[i].all_in]
        acted_since_full_raise: set[int] = set()

        while pending and self._contender_count() > 1:
            seat = pending.pop(0)
            player = self.players[seat]
            if player.folded or player.all_in:
                continue
            call_amount = max(0, self.current_bet - player.street)
            can_raise = (
                seat not in acted_since_full_raise
                and player.stack > call_amount
                and any(i != seat and not other.folded and not other.all_in
                        for i, other in enumerate(self.players))
            )
            observation = self._observation(seat, can_raise)
            action = player.bot.act(observation)
            if not isinstance(action, Action):
                raise TypeError(f"{player.name}.act() must return an Action.")

            if action.kind == "fold":
                if call_amount == 0:
                    raise ValueError(f"{player.name} folded when checking was free.")
                player.folded = True
            elif action.kind == "check":
                if call_amount:
                    raise ValueError(f"{player.name} checked while facing a bet.")
                acted_since_full_raise.add(seat)
            elif action.kind == "call":
                if call_amount == 0:
                    raise ValueError(f"{player.name} called when checking was free.")
                self._contribute(seat, min(call_amount, player.stack))
                acted_since_full_raise.add(seat)
            elif action.kind in ("bet", "raise"):
                self._apply_bet(seat, action, observation, acted_since_full_raise)
                pending = [
                    i for i in self._clockwise(seat)
                    if not self.players[i].folded and not self.players[i].all_in
                ]
            else:
                raise ValueError(f"{player.name} returned unknown action {action.kind!r}.")

    def _apply_bet(
        self,
        seat: int,
        action: Action,
        observation: Observation,
        acted_since_full_raise: set[int],
    ) -> None:
        player = self.players[seat]
        expected = "bet" if self.current_bet == 0 else "raise"
        if action.kind != expected or not observation.can_raise:
            raise ValueError(f"{player.name} made an unavailable {action.kind} action.")
        target = action.amount
        if target <= self.current_bet or target > observation.max_raise_to:
            raise ValueError(f"{player.name} chose an invalid raise target {target}.")
        if target < observation.min_raise_to and target != observation.max_raise_to:
            raise ValueError(f"{player.name} made a raise below the minimum.")

        previous_bet = self.current_bet
        self._contribute(seat, target - player.street)
        self.current_bet = target
        raise_size = target - previous_bet
        if raise_size >= self.last_raise_size:
            self.last_raise_size = raise_size
            acted_since_full_raise.clear()
        acted_since_full_raise.add(seat)

    def _award(self) -> None:
        contenders = self._contenders()
        if len(contenders) == 1:
            self.players[contenders[0]].stack += self.pot
            return

        for seat, player in enumerate(self.players):
            others = [opponent.total for i, opponent in enumerate(self.players) if i != seat]
            if not player.folded and player.total > max(others):
                refund = player.total - max(others)
                player.total -= refund
                player.stack += refund
                self.pot -= refund

        levels = sorted({player.total for player in self.players if player.total > 0})
        previous = 0
        for level in levels:
            contributors = [player for player in self.players if player.total >= level]
            side_pot = (level - previous) * len(contributors)
            eligible = [player for player in contributors if not player.folded]
            eligible_seats = [
                i for i, player in enumerate(self.players)
                if player.total >= level and not player.folded
            ]
            ranks = {
                i: omaha_rank(self.players[i].hole, self.board)
                for i in eligible_seats
            }
            best = max(ranks.values())
            winners = [i for i, rank in ranks.items() if rank == best]
            share, remainder = divmod(side_pot, len(winners))
            for seat in winners:
                self.players[seat].stack += share
            for seat in self._clockwise(self.button):
                if remainder and seat in winners:
                    self.players[seat].stack += 1
                    remainder -= 1
            previous = level


def run_match(
    first: object,
    second: object,
    *,
    hands: int = 2_000,
    starting_stack: int = 1_000,
    small_blind: int = 5,
    big_blind: int = 10,
    seed: int = 42,
) -> MatchResult:
    """Play a fixed-stack heads-up match with identical seeded deals across matches.

    Stacks reset to `starting_stack` after each hand. The button alternates every
    hand, and the same shuffled deck for hand N is reused by every round-robin match.
    """
    if hands <= 0 or starting_stack <= 0:
        raise ValueError("Hands and starting stack must be positive.")
    if small_blind <= 0 or big_blind < small_blind:
        raise ValueError("Blinds must be positive, with big blind >= small blind.")

    names = (getattr(first, "name", type(first).__name__),
             getattr(second, "name", type(second).__name__))
    players = [_Player(names[0], first, starting_stack), _Player(names[1], second, starting_stack)]
    for seat, bot in enumerate((first, second)):
        set_seed = getattr(bot, "set_seed", None)
        if callable(set_seed):
            set_seed(seed + seat * 1_000_003)

    net = [0, 0]
    hand_wins = [0, 0]
    for hand_number in range(hands):
        players[0].stack = players[1].stack = starting_stack
        button = hand_number % 2
        deck = list(DECK)
        random.Random(seed + hand_number).shuffle(deck)
        _Hand(players, button, small_blind, big_blind, deck).play()
        for seat, player in enumerate(players):
            net[seat] += player.stack - starting_stack
        if players[0].stack > players[1].stack:
            hand_wins[0] += 1
        elif players[1].stack > players[0].stack:
            hand_wins[1] += 1

    return MatchResult(names, hands, tuple(net), tuple(hand_wins))


def run_round_robin(
    competitors: Sequence[tuple[str, Callable[[], object]]],
    *,
    hands_per_match: int = 2_000,
    starting_stack: int = 1_000,
    small_blind: int = 5,
    big_blind: int = 10,
    seed: int = 42,
) -> list[Standing]:
    """Run every unordered bot pairing against the same seeded hand sequence."""
    if len(competitors) < 2:
        raise ValueError("A round robin requires at least two competitors.")
    names = [name for name, _ in competitors]
    if len(set(names)) != len(names):
        raise ValueError("Competitor names must be unique.")

    totals = {name: {"matches": 0, "wins": 0, "losses": 0, "ties": 0,
                     "net": 0, "hand_wins": 0} for name in names}
    for left in range(len(competitors)):
        for right in range(left + 1, len(competitors)):
            left_name, left_factory = competitors[left]
            right_name, right_factory = competitors[right]
            result = run_match(
                left_factory(), right_factory(), hands=hands_per_match,
                starting_stack=starting_stack, small_blind=small_blind,
                big_blind=big_blind, seed=seed,
            )
            left_stats, right_stats = totals[left_name], totals[right_name]
            left_stats["matches"] += 1
            right_stats["matches"] += 1
            left_stats["net"] += result.net_chips[0]
            right_stats["net"] += result.net_chips[1]
            left_stats["hand_wins"] += result.hand_wins[0]
            right_stats["hand_wins"] += result.hand_wins[1]
            if result.net_chips[0] > result.net_chips[1]:
                left_stats["wins"] += 1
                right_stats["losses"] += 1
            elif result.net_chips[1] > result.net_chips[0]:
                right_stats["wins"] += 1
                left_stats["losses"] += 1
            else:
                left_stats["ties"] += 1
                right_stats["ties"] += 1

    standings = [
        Standing(name, stats["matches"], stats["wins"], stats["losses"],
                 stats["ties"], stats["net"], stats["hand_wins"])
        for name, stats in totals.items()
    ]
    return sorted(standings, key=lambda row: (row.match_wins, row.net_chips), reverse=True)


def run_multi_seed_round_robin(
    competitors: Sequence[tuple[str, Callable[[], object]]],
    *,
    seeds: Sequence[int],
    hands_per_match: int = 2_000,
    starting_stack: int = 1_000,
    small_blind: int = 5,
    big_blind: int = 10,
) -> list[Standing]:
    """Aggregate round-robin records across independent seeded deal sequences."""
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    totals: dict[str, list[int]] = {}
    seed_wins: dict[str, list[int]] = {}
    seed_matches: dict[str, list[int]] = {}
    for seed in seeds:
        seed_standings = run_round_robin(
            competitors,
            hands_per_match=hands_per_match,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            seed=seed,
        )
        for standing in seed_standings:
            stats = totals.setdefault(standing.name, [0, 0, 0, 0, 0, 0])
            stats[0] += standing.matches
            stats[1] += standing.match_wins
            stats[2] += standing.match_losses
            stats[3] += standing.match_ties
            stats[4] += standing.net_chips
            stats[5] += standing.hand_wins
            seed_wins.setdefault(standing.name, []).append(standing.match_wins)
            seed_matches.setdefault(standing.name, []).append(standing.matches)

    standings = [
        Standing(
            name, *stats,
            seed_match_wins=tuple(seed_wins[name]),
            seed_match_counts=tuple(seed_matches[name]),
        )
        for name, stats in totals.items()
    ]
    return sorted(standings, key=lambda row: (row.match_wins, row.net_chips), reverse=True)


__all__ = [
    "Action", "Card", "DECK", "MatchResult", "Observation", "Standing",
    "omaha_rank", "rank_five", "run_match", "run_multi_seed_round_robin",
    "run_round_robin",
]
