"""Run the built-in bots or plug in teams' Python bots for a round robin."""

from __future__ import annotations

import argparse
import importlib
from typing import Callable

from poker_bots import (
    AdaptiveCallBot,
    BluffBot,
    BlockerBluffBot,
    BoardTextureBot,
    CallingStationBot,
    DelayedAggressorBot,
    FieldBalancedBot,
    GENERATED_STRATEGIES,
    LooseAggressiveBot,
    ManiacBot,
    PotOddsBot,
    PotControlBot,
    PositionPressureBot,
    RangePressureBot,
    RandomBot,
    RockBot,
    SPRShoveBot,
    SmallBallBot,
    ShortDeckPLOBot,
    TightAggressiveBot,
    TrapValueBot,
    ValueFirstBot,
)
from short_deck_plo import run_multi_seed_round_robin


REFERENCE_STRATEGIES = (
    ("ShortDeckPLO", ShortDeckPLOBot),
    ("Random", RandomBot),
    ("CallingStation", CallingStationBot),
    ("TightAggressive", TightAggressiveBot),
    ("LooseAggressive", LooseAggressiveBot),
    ("Rock", RockBot),
    ("Maniac", ManiacBot),
    ("PotControl", PotControlBot),
    ("Bluff", BluffBot),
    ("ValueFirst", ValueFirstBot),
    ("PotOdds", PotOddsBot),
    ("PositionPressure", PositionPressureBot),
    ("SPRShove", SPRShoveBot),
    ("TrapValue", TrapValueBot),
    ("SmallBall", SmallBallBot),
    ("BoardTexture", BoardTextureBot),
    ("RangePressure", RangePressureBot),
    ("AdaptiveCall", AdaptiveCallBot),
    ("DelayedAggressor", DelayedAggressorBot),
    ("BlockerBluff", BlockerBluffBot),
    ("FieldBalanced", FieldBalancedBot),
)
ALL_BUILTIN_STRATEGIES = (*REFERENCE_STRATEGIES, *GENERATED_STRATEGIES)


def _load_competitor(specification: str) -> tuple[str, Callable[[], object]]:
    try:
        name, target = specification.split("=", 1)
        module_name, factory_name = target.split(":", 1)
        factory = getattr(importlib.import_module(module_name), factory_name)
    except (ValueError, ImportError, AttributeError) as error:
        raise argparse.ArgumentTypeError(
            "Use NAME=python.module:BotClass (the class must implement act(observation))."
        ) from error
    if not callable(factory):
        raise argparse.ArgumentTypeError(f"{target} is not callable.")
    return name, factory


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Provide at least one unique seed.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-strategies", action="store_true",
                        help="Run the full 201-bot portfolio; defaults to one hand per matchup.")
    parser.add_argument("--list-strategies", action="store_true",
                        help="List all available built-in strategy names and exit.")
    parser.add_argument(
        "--bot", action="append", type=_load_competitor, default=[],
        metavar="NAME=MODULE:CLASS",
        help="Add an external bot class; pass multiple times. If used, only supplied bots compete.",
    )
    parser.add_argument("--hands", type=int, help="Hands per head-to-head match.")
    parser.add_argument("--stack", type=int, default=1_000, help="Starting stack, in chips, each hand.")
    parser.add_argument("--small-blind", type=int, default=5)
    parser.add_argument("--big-blind", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=_parse_seeds,
                        help="Repeat the round robin across comma-separated independent seeds.")
    args = parser.parse_args()

    if (
        len(ALL_BUILTIN_STRATEGIES) < 200
        or len({name for name, _ in ALL_BUILTIN_STRATEGIES}) != len(ALL_BUILTIN_STRATEGIES)
    ):
        raise RuntimeError("The built-in strategy portfolio must contain at least 200 uniquely named bots.")

    if args.list_strategies:
        print(f"{len(ALL_BUILTIN_STRATEGIES)} distinct built-in strategies:")
        for name, _ in ALL_BUILTIN_STRATEGIES:
            print(name)
        return
    if args.bot and args.all_strategies:
        parser.error("--bot cannot be combined with --all-strategies.")

    if args.all_strategies:
        competitors = ALL_BUILTIN_STRATEGIES
        hands = args.hands if args.hands is not None else 1
    else:
        competitors = args.bot or REFERENCE_STRATEGIES[:9]
        hands = args.hands if args.hands is not None else 2_000
    if hands <= 0:
        parser.error("--hands must be positive.")

    seeds = args.seeds or (args.seed,)
    run_options = {
        "hands_per_match": hands,
        "starting_stack": args.stack,
        "small_blind": args.small_blind,
        "big_blind": args.big_blind,
    }
    standings = run_multi_seed_round_robin(
        competitors, seeds=seeds, **run_options,
    )

    print(
        f"Short Deck 5-Card PLO round robin: {hands} hands per matchup, "
        f"{len(competitors)} strategies, seed(s) {', '.join(map(str, seeds))}"
    )
    print(
        f"{'Rank':>4}  {'Bot':<24} {'Matchup wins':>16} "
        f"{'Match W-L-T':>13} {'Seed W% range':>13} "
        f"{'Net chips':>12} {'Hand wins':>11}"
    )
    for rank, row in enumerate(standings, 1):
        record = f"{row.match_wins}-{row.match_losses}-{row.match_ties}"
        seed_range = row.seed_win_rate_range
        seed_record = (
            f"{seed_range[0]:.0%}-{seed_range[1]:.0%}"
            if seed_range is not None else "-"
        )
        print(
            f"{rank:>4}  {row.name:<24} {row.match_wins:>16}/{row.matches:<1}"
            f" {record:>13} {seed_record:>13} {row.net_chips:>12} {row.hand_wins:>11}"
        )


if __name__ == "__main__":
    main()
