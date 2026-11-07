"""Run four elimination/replacement generations, then rank every strategy.

Matches use the official rules and duplicate scoring (official_match.py);
strategies rank by official points, then average bb per deal.
"""

from __future__ import annotations

import argparse
from typing import Callable, Sequence

from poker_bots import (
    AdaptiveCallBot,
    BlockerBluffBot,
    BoardTextureBot,
    BluffBot,
    CallingStationBot,
    DelayedAggressorBot,
    FieldBalancedBot,
    LooseAggressiveBot,
    ManiacBot,
    PotOddsBot,
    PotControlBot,
    PositionPressureBot,
    RangePressureBot,
    RandomBot,
    RockBot,
    SPRShoveBot,
    ShortDeckPLOBot,
    SmallBallBot,
    TightAggressiveBot,
    TrapValueBot,
    ValueFirstBot,
)
from official_match import Standing, run_round_robin

Competitor = tuple[str, Callable[[], object]]

ORIGINAL: tuple[Competitor, ...] = (
    ("ShortDeckPLO", ShortDeckPLOBot),
    ("Random", RandomBot),
    ("CallingStation", CallingStationBot),
    ("TightAggressive", TightAggressiveBot),
    ("LooseAggressive", LooseAggressiveBot),
    ("Rock", RockBot),
    ("Maniac", ManiacBot),
    ("PotControl", PotControlBot),
    ("Bluff", BluffBot),
)

GENERATIONS: tuple[tuple[Competitor, ...], ...] = (
    (
        ("ValueFirst", ValueFirstBot),
        ("PotOdds", PotOddsBot),
        ("PositionPressure", PositionPressureBot),
    ),
    (
        ("SPRShove", SPRShoveBot),
        ("TrapValue", TrapValueBot),
        ("SmallBall", SmallBallBot),
    ),
    (
        ("BoardTexture", BoardTextureBot),
        ("RangePressure", RangePressureBot),
        ("AdaptiveCall", AdaptiveCallBot),
    ),
    (
        ("DelayedAggressor", DelayedAggressorBot),
        ("BlockerBluff", BlockerBluffBot),
        ("FieldBalanced", FieldBalancedBot),
    ),
)

ALL_COMPETITORS: tuple[Competitor, ...] = (
    *ORIGINAL,
    *GENERATIONS[0],
    *GENERATIONS[1],
    *GENERATIONS[2],
    *GENERATIONS[3],
)


def advance_field(
    active: Sequence[Competitor],
    standings: Sequence[Standing],
    newcomers: Sequence[Competitor],
) -> list[Competitor]:
    """Keep the top six by official points, then average bb per deal, and add three bots."""
    if len(newcomers) != 3 or len(active) < 4:
        raise ValueError("A generation needs at least four active bots and exactly three newcomers.")
    by_name = {standing.name: standing for standing in standings}
    if set(by_name) != {name for name, _ in active}:
        raise ValueError("Standings must contain exactly the active competitors.")
    if len({name for name, _ in (*active, *newcomers)}) != len(active) + len(newcomers):
        raise ValueError("Competitor names must be unique.")

    ordered = sorted(
        active,
        key=lambda competitor: (
            -by_name[competitor[0]].points,
            -by_name[competitor[0]].avg_bb_per_deal,
            competitor[0],
        ),
    )
    return [*ordered[:-3], *newcomers]


def print_standings(title: str, standings: Sequence[Standing]) -> None:
    print(f"\n{title}")
    multi_seed = any(len(row.seed_matches) > 1 for row in standings)
    header = f"{'Rank':>4}  {'Strategy':<24} {'Points':>12} {'W-C-L':>11}"
    if multi_seed:
        header += f" {'Seed pts range':>14}"
    print(header + f" {'Avg bb/deal':>12}")
    for rank, row in enumerate(standings, 1):
        record = f"{row.wins}-{row.chops}-{row.losses}"
        line = f"{rank:>4}  {row.name:<24} {row.points:>7.1f}/{row.matches:<4} {record:>11}"
        if multi_seed:
            seed_range = row.seed_point_rate_range
            line += f" {(f'{seed_range[0]:.0%}-{seed_range[1]:.0%}' if seed_range else '-'):>14}"
        print(line + f" {row.avg_bb_per_deal:>+12.3f}")


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Provide at least one unique seed.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deals", type=int, default=100,
                        help="Duplicate deals per matchup (each played from both seats).")
    parser.add_argument("--seed", type=int, default=42, help="Shared deal seed for all rounds.")
    parser.add_argument(
        "--final-seeds",
        type=parse_seeds,
        default=(42, 43, 44),
        help="Comma-separated deal seeds for the final all-strategy round robin.",
    )
    args = parser.parse_args()
    if args.deals <= 0:
        parser.error("--deals must be positive.")

    active = list(ORIGINAL)
    for generation, newcomers in enumerate(GENERATIONS, 1):
        standings = run_round_robin(active, deals=args.deals, seeds=(args.seed,))
        print_standings(f"Generation {generation} ({len(active)} strategies)", standings)
        next_field = advance_field(active, standings, newcomers)
        eliminated = [name for name, _ in active if name not in {n for n, _ in next_field}]
        print(f"  Retired from the next generation: {', '.join(eliminated)}")
        print(f"  Added strategies: {', '.join(name for name, _ in newcomers)}")
        active = next_field

    final = run_round_robin(ALL_COMPETITORS, deals=args.deals, seeds=args.final_seeds)
    print_standings(
        f"Final all-strategy round robin ({len(ALL_COMPETITORS)} strategies, "
        f"{len(args.final_seeds)} seeds: {', '.join(map(str, args.final_seeds))})",
        final,
    )


if __name__ == "__main__":
    main()
