"""Run four elimination/replacement generations, then rank every strategy."""

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
from short_deck_plo import (
    Standing,
    run_multi_seed_round_robin,
    run_round_robin,
)

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
    """Keep the top six by opponents beaten, then chips, and add three bots."""
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
            by_name[competitor[0]].match_wins,
            by_name[competitor[0]].net_chips,
        ),
        reverse=True,
    )
    return [*ordered[:-3], *newcomers]


def print_standings(title: str, standings: Sequence[Standing]) -> None:
    print(f"\n{title}")
    has_seed_rates = any(row.seed_win_rate_range is not None for row in standings)
    wins_label = "Matchup wins" if has_seed_rates else "Opponents beaten"
    header = (
        f"{'Rank':>4}  {'Strategy':<24} {wins_label:>16} "
        f"{'Match W-L-T':>13}"
    )
    if has_seed_rates:
        header += f" {'Seed W% range':>13}"
    print(header + f" {'Net chips':>12}")
    for rank, row in enumerate(standings, 1):
        record = f"{row.match_wins}-{row.match_losses}-{row.match_ties}"
        line = f"{rank:>4}  {row.name:<24} {row.match_wins:>13}/{row.matches:<3} {record:>13}"
        if has_seed_rates:
            seed_range = row.seed_win_rate_range
            seed_record = (
                f"{seed_range[0]:.0%}-{seed_range[1]:.0%}"
                if seed_range is not None else "-"
            )
            line += f" {seed_record:>13}"
        print(line + f" {row.net_chips:>12}")


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
    parser.add_argument("--hands", type=int, default=100, help="Paired hands per matchup.")
    parser.add_argument("--seed", type=int, default=42, help="Shared deal seed for all rounds.")
    parser.add_argument(
        "--final-seeds",
        type=parse_seeds,
        default=(42, 43, 44),
        help="Comma-separated deal seeds for the final all-strategy round robin.",
    )
    parser.add_argument("--stack", type=int, default=1_000)
    parser.add_argument("--small-blind", type=int, default=5)
    parser.add_argument("--big-blind", type=int, default=10)
    args = parser.parse_args()
    if args.hands <= 0:
        parser.error("--hands must be positive.")

    active = list(ORIGINAL)
    for generation, newcomers in enumerate(GENERATIONS, 1):
        standings = run_round_robin(
            active,
            hands_per_match=args.hands,
            starting_stack=args.stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            seed=args.seed,
        )
        print_standings(f"Generation {generation} ({len(active)} strategies)", standings)
        next_field = advance_field(active, standings, newcomers)
        eliminated = [name for name, _ in active if name not in {n for n, _ in next_field}]
        print(f"  Retired from the next generation: {', '.join(eliminated)}")
        print(f"  Added strategies: {', '.join(name for name, _ in newcomers)}")
        active = next_field

    final = run_multi_seed_round_robin(
        ALL_COMPETITORS,
        seeds=args.final_seeds,
        hands_per_match=args.hands,
        starting_stack=args.stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
    )
    print_standings(
        f"Final all-strategy round robin ({len(ALL_COMPETITORS)} strategies, "
        f"{len(args.final_seeds)} seeds: {', '.join(map(str, args.final_seeds))})",
        final,
    )


if __name__ == "__main__":
    main()
