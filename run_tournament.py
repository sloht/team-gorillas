"""Run the built-in bots or plug in teams' Python bots for a round robin.

Every match uses the official tournament rules and scoring (official_match.py):
duplicate deals from game.make_deal, both seats, verdict from the official
judge_engine.combine_halves (win 1 point, chop 0.5, loss 0). Standings rank by
points, then average bb per deal.
"""

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
import os

from official_match import NUM_DEALS, run_round_robin


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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-strategies", action="store_true",
                        help="Run the full 201-bot portfolio; defaults to one deal per matchup.")
    parser.add_argument("--list-strategies", action="store_true",
                        help="List all available built-in strategy names and exit.")
    parser.add_argument(
        "--bot", action="append", type=_load_competitor, default=[],
        metavar="NAME=MODULE:CLASS",
        help="Add an external bot class; pass multiple times. If used, only supplied bots compete.",
    )
    parser.add_argument("--deals", type=int,
                        help=f"Deals per duplicate match (each played from both seats). "
                             f"Default: {NUM_DEALS}, like the tournament (1 with --all-strategies).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=_parse_seeds,
                        help="Repeat the round robin across comma-separated independent seeds.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes (Linux/macOS). Default: 1.")
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
        deals = args.deals if args.deals is not None else 1
    else:
        competitors = args.bot or REFERENCE_STRATEGIES[:9]
        deals = args.deals if args.deals is not None else NUM_DEALS
    if deals <= 0:
        parser.error("--deals must be positive.")

    seeds = args.seeds or (args.seed,)
    standings = run_round_robin(competitors, deals=deals, seeds=seeds,
                                workers=args.workers or os.cpu_count() or 1)

    print(
        f"Short Deck PLO5 round robin (official rules and scoring): {deals} duplicate deals "
        f"per matchup, {len(competitors)} strategies, seed(s) {', '.join(map(str, seeds))}"
    )
    print(
        f"{'Rank':>4}  {'Bot':<24} {'Points':>12} {'W-C-L':>11} "
        f"{'Seed pts range':>14} {'Avg bb/deal':>12}"
    )
    for rank, row in enumerate(standings, 1):
        record = f"{row.wins}-{row.chops}-{row.losses}"
        seed_range = row.seed_point_rate_range
        seed_record = (
            f"{seed_range[0]:.0%}-{seed_range[1]:.0%}"
            if seed_range is not None and len(seeds) > 1 else "-"
        )
        print(
            f"{rank:>4}  {row.name:<24} {row.points:>7.1f}/{row.matches:<4}"
            f" {record:>11} {seed_record:>14} {row.avg_bb_per_deal:>+12.3f}"
        )


if __name__ == "__main__":
    main()
