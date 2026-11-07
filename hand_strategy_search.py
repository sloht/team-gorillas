"""Find the most effective strategy for every Short Deck PLO starting hand.

All play uses the official tournament rules (official/game.py, through
official_match.py): 80-chip stacks, blinds 1/2, fold / check-call / half-pot
raise / pot raise, official showdowns (three of a kind beats a straight).

For each starting hand, every candidate ("hero") strategy is dealt that hand and
plays heads-up against a field of opponent strategies. Each random seed fixes the
opponent's cards and the board, and each deal is played twice with the button
swapped. Every hero sees exactly the same deals (common random numbers), so the
strategies are compared on identical situations. The strategy with the highest
average net result, in official big blinds per hand, is stored as the best
strategy for that hand.

Starting hands are grouped by suit symmetry by default: the 376,992 possible
five-card hands collapse to 19,998 distinct hands, and each result applies to
every suit relabelling of that hand. Pass --all-combos to test every raw combo.

Only the top 40 strategies are compared per hand. A qualifying round robin of
all 201 strategies picks them first. Each pairing is a duplicate match scored
by the official judge (judge_engine.combine_halves): win 1 point, chop 0.5,
loss 0. Strategies rank by points, then average bb per deal.

The run is long, so it is built for a large machine:

    # 1. Qualifying round robin; writes results/qualifiers.json (resumable)
    python3 hand_strategy_search.py qualify --out results

    # 2. See how much work the per-hand search is, without playing any hands
    python3 hand_strategy_search.py estimate --out results --seeds 1-10

    # 3. Run it (uses every CPU core; safe to stop and rerun, it resumes)
    python3 hand_strategy_search.py run --out results --seeds 1-10

    #    Optionally split across machines: copy qualifiers.json to each one,
    #    run shard k of N, then copy every shard file into one results folder
    python3 hand_strategy_search.py run --out results --shard 0/4 --seeds 1-10

    # 4. Combine the shard files into the best-strategy table
    python3 hand_strategy_search.py merge --out results

    # 5. Look up the best strategy for a hand
    python3 hand_strategy_search.py lookup --out results "As Kh Qd Jc Ts"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import blake2b
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from hand_keys import (  # re-exported for existing callers
    BEST_JSON, RANK_TEXT, SUIT_TEXT, TEXT_RANK, canonical_hand, format_card,
    format_hand, hand_key, parse_hand,
)
from run_tournament import ALL_BUILTIN_STRATEGIES, REFERENCE_STRATEGIES
from official_match import BIG_BLIND, OfficialSeat, match_record, play_hand, standings_from_records
from official_rules import OFFICIAL_COMMIT, cards_to_official, evaluator, game
from short_deck_plo import DECK, Card

_STRATEGY_LOOKUP: dict[str, Callable[[], object]] = dict(ALL_BUILTIN_STRATEGIES)
STRATEGY_SETS: dict[str, tuple[str, ...]] = {
    "all": tuple(name for name, _ in ALL_BUILTIN_STRATEGIES),
    "reference": tuple(name for name, _ in REFERENCE_STRATEGIES),
    "lineup": tuple(name for name, _ in REFERENCE_STRATEGIES[:9]),
}
CONFIG_FILE = "run_config.json"
QUALIFIERS_FILE = "qualifiers.json"
QUALIFIER_CONFIG_FILE = "qualifier_config.json"
QUALIFIER_MATCHES_FILE = "qualifier_matches.jsonl"
DEFAULT_TOP = 40
BEST_CSV = "best_strategy_by_hand.csv"
SUMMARY_CSV = "strategy_summary.csv"


# --------------------------------------------------------------------------- #
# Cards and starting hands
# --------------------------------------------------------------------------- #

def starting_hands(all_combos: bool = False) -> list[tuple[tuple[Card, ...], int]]:
    """Return every starting hand to test with its number of raw combos, in a fixed order."""
    if all_combos:
        return [(tuple(sorted(hand)), 1) for hand in combinations(DECK, 5)]
    weights: dict[tuple[Card, ...], int] = defaultdict(int)
    for hand in combinations(DECK, 5):
        weights[canonical_hand(hand)] += 1
    return sorted(weights.items())


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SearchConfig:
    heroes: tuple[str, ...]
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    deals_per_seed: int
    all_combos: bool
    rules: str = f"official heads-up-bundle {OFFICIAL_COMMIT}"

    @property
    def hands_per_strategy(self) -> int:
        """Hands each hero plays with one starting hand."""
        return len(self.opponents) * len(self.seeds) * self.deals_per_seed * 2

    def comparable(self) -> dict:
        return asdict(self)


def resolve_strategies(spec: str) -> tuple[str, ...]:
    """Turn 'all', 'reference', 'lineup' or a comma list of names into strategy names."""
    if spec in STRATEGY_SETS:
        return STRATEGY_SETS[spec]
    names = tuple(part.strip() for part in spec.split(",") if part.strip())
    unknown = [name for name in names if name not in _STRATEGY_LOOKUP]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown strategies: {', '.join(unknown)}. "
            "Use 'all', 'reference', 'lineup' or names from run_tournament.py --list-strategies."
        )
    if not names or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("Give at least one strategy, without repeats.")
    return names


def parse_seeds(text: str) -> tuple[int, ...]:
    """Parse '1,2,3', '1-10' or a mix such as '1-5,42'."""
    seeds: list[int] = []
    try:
        for part in text.split(","):
            part = part.strip()
            dash = part.find("-", 1)  # skip a leading minus sign
            if dash > 0:
                low, high = int(part[:dash]), int(part[dash + 1:])
                if high < low:
                    raise ValueError
                seeds.extend(range(low, high + 1))
            elif part:
                seeds.append(int(part))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds look like '1-10' or '1,2,3'.") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Give at least one seed, without repeats.")
    return tuple(seeds)


def parse_shard(text: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Shards look like '0/4' (shard 0 of 4).") from error
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError("Shard index must be from 0 to count-1.")
    return index, count


# --------------------------------------------------------------------------- #
# Simulation of one starting hand
# --------------------------------------------------------------------------- #

def _deal_seed(seed: int, key: str, deal: int) -> int:
    digest = blake2b(f"{seed}|{key}|{deal}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def evaluate_hand(
    hand: Sequence[Card],
    config: SearchConfig,
    heroes: Optional[dict[str, object]] = None,
    opponents: Optional[dict[str, object]] = None,
) -> dict[str, list[float]]:
    """Play `hand` with every hero strategy and return {hero: [hands, sum_bb, sum_sq_bb]}.

    Every hand is an official game.Game: the hero sits in seat 0 holding `hand`,
    the opponent in seat 1 holding a random five cards, with a random board.
    Results are official big blinds (2 chips) per hand, from the hero's side.
    """
    heroes = heroes or {name: _STRATEGY_LOOKUP[name]() for name in config.heroes}
    opponents = opponents or {name: _STRATEGY_LOOKUP[name]() for name in config.opponents}
    hero_seats = {name: OfficialSeat(bot) for name, bot in heroes.items()}
    opponent_seats = {name: OfficialSeat(bot) for name, bot in opponents.items()}
    hero_hand = tuple(hand)
    key = hand_key(hero_hand)
    hero_cards = tuple(sorted(cards_to_official(hero_hand)))
    unseen = [card for card in DECK if card not in set(hero_hand)]
    stats = {name: [0, 0.0, 0.0] for name in config.heroes}

    for seed in config.seeds:
        for deal_number in range(config.deals_per_seed):
            deal_seed = _deal_seed(seed, key, deal_number)
            cards = unseen[:]
            random.Random(deal_seed).shuffle(cards)
            deal = game.Deal(
                hole_cards=(hero_cards, tuple(sorted(cards_to_official(cards[:5])))),
                board=cards_to_official(cards[5:10]),
            )
            for button in (0, 1):
                for opponent_name in config.opponents:
                    opponent = opponent_seats[opponent_name]
                    for hero_name in config.heroes:
                        hero = hero_seats[hero_name]
                        # Same bot randomness for every hero on the same deal.
                        hero.set_seed(deal_seed + button)
                        opponent.set_seed(deal_seed + button + 1_000_003)
                        finished = play_hand((hero, opponent), deal, button)
                        result = finished.result().nets[0] / BIG_BLIND
                        row = stats[hero_name]
                        row[0] += 1
                        row[1] += result
                        row[2] += result * result
    return stats


_WORKER: dict[str, object] = {}


def _init_worker(config: SearchConfig) -> None:
    _WORKER["config"] = config
    _WORKER["heroes"] = {name: _STRATEGY_LOOKUP[name]() for name in config.heroes}
    _WORKER["opponents"] = {name: _STRATEGY_LOOKUP[name]() for name in config.opponents}


def _work(task: tuple[tuple[Card, ...], int]) -> dict:
    hand, weight = task
    config: SearchConfig = _WORKER["config"]  # type: ignore[assignment]
    stats = evaluate_hand(hand, config, _WORKER["heroes"], _WORKER["opponents"])  # type: ignore[arg-type]
    return {
        "hand": hand_key(hand),
        "weight": weight,
        "stats": {name: [n, round(total, 6), round(squares, 6)]
                  for name, (n, total, squares) in stats.items()},
    }


# --------------------------------------------------------------------------- #
# Qualifying tournament: pick the top strategies before the per-hand search
# --------------------------------------------------------------------------- #

def _qualify_work(task: tuple) -> dict:
    left, right, seed, deals = task
    return match_record(left, right, _STRATEGY_LOOKUP[left](), _STRATEGY_LOOKUP[right](),
                        seed, deals)


def rank_qualifiers(matches: Iterable[dict], pool: Sequence[str]) -> list[dict]:
    """Rank like the official tournament: points (win 1, chop 0.5, loss 0), then
    average bb per deal over all matches, then name.

    Each match is one duplicate match (one pairing on one seed), with the
    verdict from the official judge_engine.combine_halves().
    """
    return [
        {"name": row.name, "matches": row.matches, "points": row.points, "wins": row.wins,
         "chops": row.chops, "losses": row.losses,
         "avg_bb_per_deal": round(row.avg_bb_per_deal, 6)}
        for row in standings_from_records(matches, pool)
    ]


def load_qualified(out: str | Path, path: Optional[str | Path] = None) -> tuple[str, ...]:
    """Read the saved top strategies from --qualifiers, or from qualifiers.json in --out."""
    path = Path(path) if path else Path(out) / QUALIFIERS_FILE
    if not path.exists():
        raise SystemExit(
            f"No {path}. Run the qualifying tournament first:\n"
            f"  python3 hand_strategy_search.py qualify --out {out}\n"
            "or choose heroes yourself with --heroes all|reference|lineup|NAME,NAME."
        )
    return tuple(json.loads(path.read_text())["qualified"])


def command_qualify(args: argparse.Namespace) -> None:
    pool = args.pool
    if args.top < 1:
        raise SystemExit("--top must be at least 1.")
    if args.deals < 1:
        raise SystemExit("--deals must be at least 1.")
    settings = {"pool": list(pool), "seeds": list(args.seeds), "deals_per_match": args.deals,
                "scoring": "official duplicate (judge_engine.combine_halves)",
                "rules": f"official heads-up-bundle {OFFICIAL_COMMIT}"}
    pairings = [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i + 1, len(pool))]
    total_matches = len(pairings) * len(args.seeds)
    print(f"Qualifying: {len(pool)} strategies, {len(pairings):,} pairings x "
          f"{len(args.seeds)} seeds = {total_matches:,} duplicate matches of "
          f"{args.deals} deals x 2 seats = {total_matches * args.deals * 2:,} poker hands; "
          f"keeping the top {args.top}.", file=sys.stderr)
    if args.estimate_only:
        return

    out = Path(args.out)
    saved_path = out / QUALIFIERS_FILE
    if saved_path.exists() and not args.force:
        saved = json.loads(saved_path.read_text())
        if saved.get("config") == settings and saved.get("top") == args.top:
            print(f"Already qualified; using the saved list in {saved_path} "
                  "(delete it to run the tournament again):")
            for place, name in enumerate(saved["qualified"], 1):
                print(f"  {place:>3}. {name}")
            return
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / QUALIFIER_CONFIG_FILE
    if config_path.exists():
        if json.loads(config_path.read_text()) != settings and not args.force:
            raise SystemExit(f"{config_path} was written with different settings. Use a new "
                             "--out folder, or pass --force (not recommended).")
    else:
        config_path.write_text(json.dumps(settings, indent=2))

    matches_path = out / QUALIFIER_MATCHES_FILE
    finished: dict[tuple, dict] = {}
    if matches_path.exists():
        with matches_path.open() as handle:
            for line in handle:
                try:
                    match = json.loads(line)
                    finished[(match["left"], match["right"], match["seed"])] = match
                except (json.JSONDecodeError, KeyError):
                    continue
    tasks = [(left, right, seed, args.deals)
             for seed in args.seeds for left, right in pairings
             if (left, right, seed) not in finished]
    print(f"  {len(finished):,} matches already done, {len(tasks):,} to go.", file=sys.stderr)

    if tasks:
        workers = args.workers or os.cpu_count() or 1
        evaluator.warmup()  # build the official table once; forked workers share it
        started = time.time()
        with matches_path.open("a") as handle:
            pool_ = None
            if workers == 1:
                results = map(_qualify_work, tasks)
            else:
                pool_ = mp.get_context("spawn" if sys.platform == "win32" else "fork").Pool(workers)
                results = pool_.imap_unordered(_qualify_work, tasks, chunksize=4)
            try:
                for count, match in enumerate(results, 1):
                    handle.write(json.dumps(match, separators=(",", ":")) + "\n")
                    handle.flush()
                    finished[(match["left"], match["right"], match["seed"])] = match
                    if count % args.progress_every == 0 or count == len(tasks):
                        elapsed = time.time() - started
                        left_h = elapsed / count * (len(tasks) - count) / 3600
                        print(f"  {count:,}/{len(tasks):,} matches, ~{left_h:.2f} h left",
                              file=sys.stderr)
            finally:
                if pool_ is not None:
                    pool_.terminate()
                    pool_.join()

    wanted = {(left, right, seed) for seed in args.seeds for left, right in pairings}
    ranking = rank_qualifiers((m for key, m in finished.items() if key in wanted), pool)
    qualified = [row["name"] for row in ranking[:args.top]]
    (out / QUALIFIERS_FILE).write_text(json.dumps(
        {"config": settings, "top": args.top, "qualified": qualified, "ranking": ranking},
        indent=1))
    print(f"Top {len(qualified)} written to {out / QUALIFIERS_FILE}:")
    for place, row in enumerate(ranking[:args.top], 1):
        print(f"  {place:>3}. {row['name']:<24} {row['points']:>6.1f} points from "
              f"{row['matches']} matches (W{row['wins']} C{row['chops']} L{row['losses']}), "
              f"{row['avg_bb_per_deal']:+.3f} bb/deal")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def _resolve_heroes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.heroes != "qualified":
        return resolve_strategies(args.heroes)
    path = Path(args.qualifiers) if args.qualifiers else Path(args.out) / QUALIFIERS_FILE
    if args.command == "estimate" and not path.exists():
        print(f"(No {path} yet; estimating with {DEFAULT_TOP} qualified heroes.)")
        return STRATEGY_SETS["all"][:DEFAULT_TOP]
    return load_qualified(args.out, path)


def _config_from_args(args: argparse.Namespace) -> SearchConfig:
    if args.deals_per_seed < 1:
        raise SystemExit("--deals-per-seed must be at least 1.")
    return SearchConfig(
        heroes=_resolve_heroes(args),
        opponents=args.opponents,
        seeds=args.seeds,
        deals_per_seed=args.deals_per_seed,
        all_combos=args.all_combos,
    )


def _shard_tasks(args: argparse.Namespace, config: SearchConfig) -> list[tuple[tuple[Card, ...], int]]:
    tasks = starting_hands(config.all_combos)
    index, count = args.shard
    tasks = tasks[index::count]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    return tasks


def command_estimate(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    tasks = _shard_tasks(args, config)
    per_hand = config.hands_per_strategy * len(config.heroes)
    total = per_hand * len(tasks)
    print(f"Starting hands to test : {len(tasks):,}"
          f"{' (raw combos)' if config.all_combos else ' (suit-distinct)'}")
    print(f"Hero strategies        : {len(config.heroes)}")
    print(f"Opponent strategies    : {len(config.opponents)}")
    print(f"Seeds x deals x 2 seats: {len(config.seeds)} x {config.deals_per_seed} x 2")
    print(f"Hands per hero per hand: {config.hands_per_strategy:,}")
    print(f"Poker hands in total   : {total:,}")
    if args.ms_per_hand:
        cpus = args.workers or os.cpu_count() or 1
        hours = total * args.ms_per_hand / 1000 / 3600 / cpus
        print(f"At {args.ms_per_hand} ms/hand on {cpus} cores: about {hours:,.1f} hours")


def command_run(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config_path = out / CONFIG_FILE
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != json.loads(json.dumps(config.comparable())) and not args.force:
            raise SystemExit(
                f"{config_path} was written with different settings. Use a new --out "
                "folder, or pass --force to mix results (not recommended)."
            )
    else:
        config_path.write_text(json.dumps(config.comparable(), indent=2))

    index, count = args.shard
    shard_path = out / f"shard-{index:03d}-of-{count:03d}.jsonl"
    done: set[str] = set()
    if shard_path.exists():
        with shard_path.open() as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["hand"])
                except (json.JSONDecodeError, KeyError):
                    continue  # a line cut off by an interrupted run is redone
    tasks = [task for task in _shard_tasks(args, config) if hand_key(task[0]) not in done]
    print(f"Shard {index}/{count}: {len(done):,} hands already done, {len(tasks):,} to go "
          f"({len(config.heroes)} heroes x {config.hands_per_strategy:,} hands each).",
          file=sys.stderr)
    if not tasks:
        return

    workers = args.workers or os.cpu_count() or 1
    evaluator.warmup()  # build the official table once; forked workers share it
    started = time.time()
    with shard_path.open("a") as handle:
        if workers == 1:
            _init_worker(config)
            results = map(_work, tasks)
            pool = None
        else:
            pool = mp.get_context("spawn" if sys.platform == "win32" else "fork").Pool(
                workers, initializer=_init_worker, initargs=(config,)
            )
            results = pool.imap_unordered(_work, tasks, chunksize=1)
        try:
            for finished, record in enumerate(results, 1):
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                if finished % args.progress_every == 0 or finished == len(tasks):
                    elapsed = time.time() - started
                    remaining = elapsed / finished * (len(tasks) - finished)
                    print(f"  {finished:,}/{len(tasks):,} hands, "
                          f"{elapsed / 60:.1f} min elapsed, ~{remaining / 3600:.2f} h left",
                          file=sys.stderr)
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()


def _load_records(out: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(out.glob("shard-*.jsonl")):
        with path.open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.setdefault(record["hand"], record)
    return records


def summarize_hand(stats: dict[str, Sequence[float]]) -> dict:
    """Rank strategies for one hand by mean big blinds per hand."""
    rows = []
    for name, (n, total, squares) in stats.items():
        if n == 0:
            continue
        mean = total / n
        variance = max(0.0, squares / n - mean * mean) * n / max(1, n - 1)
        rows.append((mean, math.sqrt(variance / n), name))
    rows.sort(key=lambda row: (-row[0], row[2]))
    best_mean, best_se, best = rows[0]
    summary = {"best": best, "best_bb_per_hand": round(best_mean, 4),
               "best_std_error": round(best_se, 4), "hands_per_strategy": int(stats[best][0]),
               "runner_up": None, "runner_up_bb_per_hand": None, "margin": None,
               "clear_winner": True}
    if len(rows) > 1:
        second_mean, second_se, second = rows[1]
        margin = best_mean - second_mean
        # Conservative: deals are shared, so the true error of the gap is smaller.
        summary.update(runner_up=second, runner_up_bb_per_hand=round(second_mean, 4),
                       margin=round(margin, 4),
                       clear_winner=margin > 2 * math.hypot(best_se, second_se))
    return summary


def command_merge(args: argparse.Namespace) -> None:
    out = Path(args.out)
    records = _load_records(out)
    if not records:
        raise SystemExit(f"No shard-*.jsonl results found in {out}.")
    config = json.loads((out / CONFIG_FILE).read_text()) if (out / CONFIG_FILE).exists() else {}
    expected = len(starting_hands(config.get("all_combos", False))) if config else None

    table: dict[str, dict] = {}
    best_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    overall: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for key in sorted(records):
        record = records[key]
        summary = summarize_hand(record["stats"])
        summary["combos"] = record["weight"]
        table[key] = summary
        best_counts[summary["best"]][0] += 1
        best_counts[summary["best"]][1] += record["weight"]
        for name, (n, total, _) in record["stats"].items():
            overall[name][0] += total / n * record["weight"] if n else 0.0
            overall[name][1] += record["weight"]

    (out / BEST_JSON).write_text(json.dumps(
        {"config": config, "suit_symmetry": not config.get("all_combos", False),
         "hands": table}, indent=1))
    columns = ["hand", "combos", "best", "best_bb_per_hand", "best_std_error",
               "runner_up", "runner_up_bb_per_hand", "margin", "clear_winner",
               "hands_per_strategy"]
    with (out / BEST_CSV).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key, summary in table.items():
            writer.writerow({"hand": key, **{c: summary[c] for c in columns[1:]}})

    total_combos = sum(record["weight"] for record in records.values())
    with (out / SUMMARY_CSV).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strategy", "hands_where_best", "share_of_all_combos",
                         "avg_bb_per_hand_all_hands"])
        for name in sorted(overall, key=lambda n: -best_counts.get(n, (0, 0))[1]):
            wins, combos = best_counts.get(name, (0, 0))
            total, weight = overall[name]
            writer.writerow([name, wins, round(combos / total_combos, 6),
                             round(total / weight, 4) if weight else ""])

    clear = sum(summary["clear_winner"] for summary in table.values())
    print(f"Merged {len(table):,} hands"
          + (f" of {expected:,}" if expected else "")
          + f" into {out / BEST_CSV} and {out / BEST_JSON}.")
    print(f"Clear winner (gap > 2 std errors) in {clear:,} hands; "
          "the rest are too close to call with this many seeds.")
    print("Most often best:")
    for name, (wins, combos) in sorted(best_counts.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {name:<24} {wins:>7,} hands  {combos / total_combos:6.1%} of combos")


def load_best_table(path: str | Path) -> dict:
    """Load a merged table (results folder or best_strategy_by_hand.json)."""
    path = Path(path)
    if path.is_dir():
        path = path / BEST_JSON
    return json.loads(path.read_text())


def best_strategy_for(hole: Sequence[Card] | str, table: dict) -> Optional[dict]:
    """Return the stored result for a five-card hand, or None if it was not tested."""
    hand = parse_hand(hole) if isinstance(hole, str) else tuple(hole)
    lookup = hand if not table.get("suit_symmetry", True) else canonical_hand(hand)
    return table["hands"].get(hand_key(lookup))


def command_lookup(args: argparse.Namespace) -> None:
    table = load_best_table(args.out)
    for text in args.hands:
        result = best_strategy_for(text, table)
        if result is None:
            print(f"{text}: not in the results yet")
            continue
        line = (f"{text}: best is {result['best']} at {result['best_bb_per_hand']:+.3f} bb/hand")
        if result["runner_up"]:
            line += (f"; runner-up {result['runner_up']} at "
                     f"{result['runner_up_bb_per_hand']:+.3f}"
                     f" ({'clear winner' if result['clear_winner'] else 'too close to call'})")
        print(line)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def simulation_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--heroes", default="qualified",
                             help="Strategies to compare: qualified (the top strategies from "
                                  "'qualify', read from --out), all (201), reference (21), "
                                  "lineup (9), or comma-separated names. Default: qualified.")
        command.add_argument("--out", default="hand_strategy_results", help="Results folder.")
        command.add_argument("--qualifiers",
                             help="A saved qualifiers.json to read the top strategies from. "
                                  "Default: qualifiers.json in --out.")
        command.add_argument("--opponents", type=resolve_strategies, default="lineup",
                             help="Field each hero plays against; same choices. Default: lineup "
                                  "(the 9 bots in the default tournament).")
        command.add_argument("--seeds", type=parse_seeds, default="1-5",
                             help="Random seeds, e.g. '1-10' or '1,2,42'. Default: 1-5.")
        command.add_argument("--deals-per-seed", type=int, default=2,
                             help="Different opponent hands and boards per seed. Default: 2.")
        command.add_argument("--all-combos", action="store_true",
                             help="Test all 376,992 raw hands instead of 19,998 suit-distinct ones.")
        command.add_argument("--shard", type=parse_shard, default=(0, 1),
                             help="Run only shard K of N, written 'K/N'. Default: 0/1 (everything).")
        command.add_argument("--limit", type=int,
                             help="Only the first N hands of this shard (for a quick trial).")
        command.add_argument("--workers", type=int,
                             help="Worker processes. Default: every CPU core.")

    estimate = commands.add_parser("estimate", help="Count the work without playing any hands.")
    simulation_options(estimate)
    estimate.add_argument("--ms-per-hand", type=float,
                          help="Measured milliseconds per poker hand, to estimate run time.")
    estimate.set_defaults(func=command_estimate)

    qualify = commands.add_parser(
        "qualify", help="Round robin of all strategies; keep the top N for the per-hand search.")
    qualify.add_argument("--pool", type=resolve_strategies, default="all",
                         help="Strategies that compete to qualify. Default: all (201).")
    qualify.add_argument("--top", type=int, default=DEFAULT_TOP,
                         help=f"How many qualify. Default: {DEFAULT_TOP}.")
    qualify.add_argument("--seeds", type=parse_seeds, default="1001-1003",
                         help="Round-robin seeds; keep them apart from the search seeds. "
                              "Default: 1001-1003.")
    qualify.add_argument("--deals", type=int, default=200,
                         help="Deals per duplicate match; each is played from both seats. "
                              "The tournament uses 1000. Default: 200.")
    qualify.add_argument("--out", default="hand_strategy_results", help="Results folder.")
    qualify.add_argument("--workers", type=int, help="Worker processes. Default: every CPU core.")
    qualify.add_argument("--force", action="store_true",
                         help="Allow resuming into a folder made with different settings.")
    qualify.add_argument("--estimate-only", action="store_true",
                         help="Print the amount of work and stop without playing.")
    qualify.add_argument("--progress-every", type=int, default=500)
    qualify.set_defaults(func=command_qualify)

    run = commands.add_parser("run", help="Play the simulation and write shard results.")
    simulation_options(run)
    run.add_argument("--force", action="store_true",
                     help="Allow resuming into a folder made with different settings.")
    run.add_argument("--progress-every", type=int, default=25)
    run.set_defaults(func=command_run)

    merge = commands.add_parser("merge", help="Combine shard files into the best-strategy table.")
    merge.add_argument("--out", default="hand_strategy_results", help="Results folder.")
    merge.set_defaults(func=command_merge)

    lookup = commands.add_parser("lookup", help="Show the best strategy for given hands.")
    lookup.add_argument("--out", default="hand_strategy_results", help="Results folder.")
    lookup.add_argument("hands", nargs="+", help="Hands such as 'As Kh Qd Jc Ts'.")
    lookup.set_defaults(func=command_lookup)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
