"""Keep generating strategies and replace the champion when one is better against a broad field.

    python champion_search.py                       # run until stopped (Ctrl+C); rerun to resume
    python champion_search.py --generations 3       # stop after 3 more generations
    python champion_search.py --status              # print the champion and history, play nothing
    python champion_search.py --out evolution_b --lineage B --start random --peer evolution
                                                    # a second, independent search line
    python champion_search.py --final --peer evolution_b
                                                    # both champions vs every known bot + each other

Every --check-every generations (default 5) the champion plays every bot ever
seen, on a fresh seed: all built-in strategies, the practice bots, every past
champion and candidate of this lineage and of each --peer lineage. The result
is printed in a box of asterisks and saved to <out>/periodic_checks.jsonl.

Every strategy is an evolved_policy.Genome. The first champion is PotShove08
(the default genome, which plays exactly like poker_bots.PotShove08).

Guarding against overfitting. A new champion has to be better against MANY
opponents, not just good against the current champion:

* The TRAINING field has 22 bots of every style (strong qualifiers, tight,
  loose, passive, maniacs, random, the bundle's always-half and always-pot
  practice bots) plus EVERY past champion. Selection is driven by results
  against this field. Beating the champion head to head only counts for a
  quarter of the screening score.
* A separate HOLDOUT field of 12 other bots is never used to pick candidates.
  It only checks, at confirmation, that an improvement carries over to
  opponents the search did not tune against.
* Confirmation replays the best screened candidates on fresh seeds.

Each generation:

1. Make --candidates new genomes: small and larger mutations of the champion,
   mutations and crossovers of the best recent candidates, and some random
   genomes from the whole search space.
2. Screen on one seed: each candidate plays one official duplicate match
   (1,000 deals, both seats) against every training-field bot, and two against
   the champion. The champion plays the same field on the same seed.
   Score = mean over opponents of (candidate bb/deal - champion bb/deal)
   + 0.25 x candidate's bb/deal against the champion. Candidates much slower
   than the champion are dropped (the tournament allows ~8 s per 1,000 hands).
3. Confirm the best few positive scores on 2 fresh seeds. A candidate becomes
   champion only if ALL of these hold:
   a. against the training field, its per-opponent advantage over the champion
      is more than 1.645 standard errors above 0 (one-sided 95%);
   b. it does better than the champion against at least 60% of the training
      opponents (breadth, not a few big wins);
   c. it gets at least the champion's tournament points against the training field;
   d. against the holdout field it is not worse than the champion (bb/deal) and
      gets at least the champion's points minus 1;
   e. it does not lose to the champion head to head (mean over 6 matches >= 0).
   The old champion then joins the training field.

Matches are official (official_match.match_record, scored by the judge's
combine_halves). Every finished match is appended to <out>/matches.jsonl and
reused on a rerun; generation results go to <out>/candidates.jsonl and the
current state to <out>/state.json and <out>/champion.json.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import statistics as st
import sys
import time
from dataclasses import replace
from pathlib import Path

from bot_similarity import FingerprintCache, dedupe, fingerprint
from diverse_bots import DIVERSE_BOTS
from evolved_policy import (GENE_RANGES, POTSHOVE08, EvolvedPolicy, Genome, crossover, mutate,
                            use_gene_ranges,
                            random_genome)
from official_match import match_record, standings_from_records
from official_rules import evaluator, game
from run_tournament import ALL_BUILTIN_STRATEGIES


class FixedRaiseBot:
    """The bundle's baseline_always_half / baseline_always_pot: always that raise, else check/call."""

    plays_official = True

    def __init__(self, action: "game.Action") -> None:
        self.action = action
        self.name = "AlwaysHalf" if action == game.Action.RAISE_HALF else "AlwaysPot"

    def set_seed(self, seed: int) -> None:
        pass

    def act(self, obs) -> "game.Action":
        return self.action if self.action in obs.legal_actions else game.Action.CHECK_CALL


BUILTINS = dict(ALL_BUILTIN_STRATEGIES)
BUILTINS["AlwaysHalf"] = lambda: FixedRaiseBot(game.Action.RAISE_HALF)
BUILTINS["AlwaysPot"] = lambda: FixedRaiseBot(game.Action.RAISE_POT)
BUILTINS.update(DIVERSE_BOTS)  # rule-based, line-based, mixed and adaptive opponents

# Opponents selection is tuned against. PotShove08 itself is the first
# champion, so it joins only once it is dethroned.
TRAIN_FIELD = [
    "AggressiveValue08", "PotShove04", "PotShove02", "BlockerPressure08", "AggressiveValue10",
    "PotShove07", "TightValue04", "SmallBall03", "BluffPressure08", "DefensiveCaller08",
    "TextureAware07", "ShortDeckPLO", "TightAggressive", "LooseAggressive", "CallingStation",
    "Maniac", "Rock", "Random", "PotControl", "Bluff", "AlwaysHalf", "AlwaysPot",
    # structurally different (diverse_bots.py): rules, traps, lines, mixing, adapting
    "RuleTight", "CheckRaiser", "Floater", "MixedStrategy", "NitFolder", "PreflopJammer", "Exploiter",
]
# Never used to choose candidates; only to check that gains carry over.
HOLDOUT_FIELD = [
    "AggressiveValue06", "PotShove10", "BlockerPressure04", "BluffPressure10", "EquityValue10",
    "TightValue01", "UncertaintyAware08", "LateStreet07", "BalancedField07", "ValueFirst",
    "SPRShove", "TrapValue",
    "RuleLoose", "DelayedProbe", "MixedLoose", "ExploiterAggro",
]
# Bots agreeing on at least this share of the fingerprint situations count as
# near-duplicates; only the first is kept (champions are always kept).
SIMILARITY = 0.95
DEALS = 1000
H2H_WEIGHT = 0.25
CONFIRM_H2H_SEEDS = 6
CONFIRM_FIELD_SEEDS = 2
Z_ONE_SIDED_95 = 1.645
BREADTH = 0.60       # must beat the champion's result against this share of opponents
SPEED_LIMIT = 1.6    # at most this many times the champion's ms per decision
TABU_COSINE = 0.25   # after a revert, redraw mutations this aligned with the abandoned path


# --------------------------------------------------------------------------- #
# Playing matches in worker processes
# --------------------------------------------------------------------------- #

class TimedPolicy(EvolvedPolicy):
    """EvolvedPolicy that adds up its own thinking time."""

    def __init__(self, *args, **kwargs) -> None:
        self.think = 0.0
        self.decisions = 0
        super().__init__(*args, **kwargs)

    def act(self, obs, samples=None):
        start = time.perf_counter()
        action = super().act(obs, samples)
        self.think += time.perf_counter() - start
        self.decisions += 1
        return action


def _player(spec: dict):
    if spec["kind"] == "builtin":
        return BUILTINS[spec["id"]]()
    return TimedPolicy(Genome.from_dict(spec["genome"]), name=spec["id"])


def fingerprint_work(spec: dict) -> tuple[str, str]:
    return spec["id"], fingerprint(_player(spec))


def distinct(runner: "Runner", out: Path, specs: dict[str, dict],
             keep_always: set[str] = frozenset()) -> tuple[list[str], dict[str, str]]:
    """Drop near-duplicate bots (see bot_similarity). Priority when two match:
    always-kept bots, the diverse bots, the training then holdout fields, other
    built-ins, then evolved candidates by id."""
    prints = FingerprintCache(out / "fingerprints.json").ensure(specs, fingerprint_work, runner.pool)
    preferred = [*sorted(keep_always), *DIVERSE_BOTS, *TRAIN_FIELD, *HOLDOUT_FIELD, *BUILTINS]
    order = list(dict.fromkeys([b for b in preferred if b in specs] + sorted(specs)))
    return dedupe(order, prints, SIMILARITY, set(keep_always))


def play(task: tuple) -> dict:
    left, right, seed, deals = task
    a, b = _player(left), _player(right)
    record = match_record(left["id"], right["id"], a, b, seed, deals)
    for side, player in (("left_ms", a), ("right_ms", b)):
        if isinstance(player, TimedPolicy) and player.decisions:
            record[side] = round(player.think * 1000 / player.decisions, 4)
    return record


class Runner:
    """Plays matches in parallel, caching every result in matches.jsonl."""

    def __init__(self, out: Path, workers: int) -> None:
        self.path = out / "matches.jsonl"
        self.cache: dict[tuple, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.cache[(r["left"], r["right"], r["seed"], r["deals"])] = r
                except (json.JSONDecodeError, KeyError):
                    continue
        self.pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None

    def run(self, tasks: list[tuple], label: str) -> dict[tuple, dict]:
        key = lambda t: (t[0]["id"], t[1]["id"], t[2], t[3])  # noqa: E731
        todo = list({key(t): t for t in tasks if key(t) not in self.cache}.values())
        started = time.time()
        if todo:
            results = self.pool.imap_unordered(play, todo) if self.pool else map(play, todo)
            with self.path.open("a") as handle:
                for count, record in enumerate(results, 1):
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    self.cache[(record["left"], record["right"], record["seed"],
                                record["deals"])] = record
                    if count % 150 == 0:
                        print(f"    {label}: {count}/{len(todo)} matches, "
                              f"{(time.time() - started) / 60:.1f} min", flush=True)
        print(f"    {label}: {len(tasks)} matches ({len(tasks) - len(todo)} cached) in "
              f"{(time.time() - started) / 60:.1f} min", flush=True)
        return {key(t): self.cache[key(t)] for t in tasks}

    def close(self) -> None:
        if self.pool:
            self.pool.terminate()
            self.pool.join()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def field_rows(results: dict, spec: dict, field: list[dict], seeds: list[int]) -> dict:
    """{(opponent, seed): match record} with `spec` as the left player."""
    return {(opp["id"], seed): results[(spec["id"], opp["id"], seed, DEALS)]
            for seed in seeds for opp in field}


def field_summary(rows: dict) -> dict:
    ms = [r["left_ms"] for r in rows.values() if "left_ms" in r]
    return {"points": sum(r["points"][0] for r in rows.values()), "matches": len(rows),
            "bb": st.mean(r["mean_bb"] for r in rows.values()), "ms": st.mean(ms) if ms else None}


def compare(candidate: dict, champion: dict, field: list[dict]) -> dict:
    """Paired comparison on the same opponents and seeds."""
    diffs = [candidate[k]["mean_bb"] - champion[k]["mean_bb"] for k in candidate]
    by_opp: dict[str, list[float]] = {}
    for (opp, _), c in candidate.items():
        by_opp.setdefault(opp, []).append(c["mean_bb"] - champion[(opp, _)]["mean_bb"])
    better = sum(st.mean(v) > 0 for v in by_opp.values())
    se = st.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("inf")
    return {"diff": st.mean(diffs), "se": se, "better_share": better / len(by_opp),
            "better": better, "opponents": len(by_opp)}


def head_to_head(results: dict, spec: dict, champion: dict, seeds: list[int]) -> dict:
    rows = [results[(spec["id"], champion["id"], seed, DEALS)] for seed in seeds]
    mean = st.mean(r["mean_bb"] for r in rows)
    se = math.sqrt(sum(((r["ci"][1] - r["ci"][0]) / 3.92) ** 2 for r in rows)) / len(rows)
    return {"bb": mean, "se": se, "verdicts": [r["verdict"] for r in rows]}


def _short(verdicts: list[str]) -> str:
    return "".join({"AGENT1_WIN": "W", "AGENT2_WIN": "L"}.get(v, "C") for v in verdicts)


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #

def genome_spec(name: str, genome: Genome) -> dict:
    return {"id": name, "kind": "genome", "genome": genome.to_dict()}


def load_state(out: Path, lineage: str = "", start: str = "potshove08") -> dict:
    """Resume from state.json, or start a new lineage (its settings are then kept in the state)."""
    path = out / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    if start == "random":
        champion = genome_spec(f"{lineage}start", random_genome(random.Random(f"start-{lineage}")))
    else:
        champion = genome_spec("PotShove08", POTSHOVE08)
    return {"generation": 0, "champion": champion, "hall_of_fame": [], "history": [],
            "candidates_tried": 0, "lineage": lineage, "start": start}


def seed_offset(lineage: str) -> int:
    """Lineage '' (the original run) keeps its seeds; any other lineage plays different deals."""
    return 0 if not lineage else 1_000_000 * (sum(map(ord, lineage)) % 97 + 1)


def save_state(out: Path, state: dict) -> None:
    tmp = out / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(out / "state.json")
    (out / "champion.json").write_text(json.dumps(state["champion"], indent=1))


def recent_best(out: Path, generations: int = 3, count: int = 8, since: int = 0) -> list[Genome]:
    """The best-scoring recent candidates, ignoring any from before generation `since`
    (after a revert, the abandoned branch's candidates are left out)."""
    path = out / "candidates.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["generation"] > since]
    if not rows:
        return []
    last = max(r["generation"] for r in rows)
    rows = [r for r in rows if r["generation"] > last - generations and not r.get("too_slow")]
    rows.sort(key=lambda r: -r["score"])
    return [Genome.from_dict(r["genome"]) for r in rows[:count]]


def direction(a: Genome, b: Genome) -> dict[str, float]:
    """The move from genome a to genome b, each gene scaled by its range."""
    return {name: (getattr(b, name) - getattr(a, name)) / (high - low)
            for name, (low, high) in GENE_RANGES.items()}


def cosine(u: dict[str, float], v: dict[str, float]) -> float:
    dot = sum(u[k] * v[k] for k in u)
    norm = math.sqrt(sum(x * x for x in u.values())) * math.sqrt(sum(x * x for x in v.values()))
    return dot / norm if norm else 0.0


def make_candidates(generation: int, champion: Genome, pool: list[Genome], count: int,
                    lineage: str = "", tabu: list[dict[str, float]] | None = None):
    """New genomes for one generation. Champion-derived genomes that move in a
    `tabu` direction (cosine > TABU_COSINE, a path a revert abandoned) are redrawn."""
    rng = random.Random(f"{lineage}-generation-{generation}" if lineage else f"generation-{generation}")
    made: list[tuple[str, Genome]] = []
    plan = [("small mutation", 0.30), ("large mutation", 0.25),
            ("pool mutation", 0.20 if pool else 0.0), ("crossover", 0.12 if pool else 0.0)]
    rejected = 0
    while len(made) < count:
        roll, origin = rng.random(), "random"
        for name, share in plan:
            if roll < share:
                origin = name
                break
            roll -= share
        if origin == "small mutation":
            genome = mutate(champion, rng, rate=0.25, scale=0.06)
        elif origin == "large mutation":
            genome = mutate(champion, rng, rate=0.4, scale=0.15)
        elif origin == "pool mutation":
            genome = mutate(rng.choice(pool), rng, rate=0.3, scale=0.08)
        elif origin == "crossover":
            genome = mutate(crossover(champion, rng.choice(pool), rng), rng, rate=0.1, scale=0.04)
        else:
            genome = random_genome(rng)
        if (tabu and origin != "random" and rejected < 50 * count
                and any(cosine(direction(champion, genome), d) > TABU_COSINE for d in tabu)):
            rejected += 1
            continue
        if genome != champion and all(genome != g for _, g in made):
            made.append((origin, genome))
    return [(f"{lineage}g{generation:04d}c{i:02d}", origin, genome)
            for i, (origin, genome) in enumerate(made)]


ADAPTIVE_GENES = ("a_bluff", "a_thin", "a_catch")
# One candidate per adaptation level each generation: (label, share of each gene's range).
ADAPTIVE_LEVELS = (("light", 0.05, 0.20), ("mild", 0.20, 0.40), ("moderate", 0.40, 0.60),
                   ("strong", 0.60, 0.80), ("very strong", 0.80, 1.00))
ADAPTIVE_PER_GENERATION = len(ADAPTIVE_LEVELS)


def adaptive_candidates(generation: int, champion: Genome, count: int, lineage: str = "",
                        start_index: int = 0):
    """The champion with adaptation at every level from light to very strong: for
    candidate i, each chosen adaptation gene (each with probability 0.7, at least
    one) is set within level i's share of its range, then a small mutation of the
    other genes. Mutation alone only nudges these genes up from 0, so this tests
    which amount of adaptation, if any, pays."""
    rng = random.Random(f"{lineage}-adaptive-{generation}")
    made = []
    while len(made) < count:
        label, lo, hi = ADAPTIVE_LEVELS[len(made) % len(ADAPTIVE_LEVELS)]
        chosen = [g for g in ADAPTIVE_GENES if rng.random() < 0.7] or [rng.choice(ADAPTIVE_GENES)]
        values = {}
        for gene in chosen:
            low, high = GENE_RANGES[gene]
            values[gene] = round(low + rng.uniform(lo, hi) * (high - low), 4)
        genome = mutate(replace(champion, **values), rng, rate=0.1, scale=0.04)
        genome = replace(genome, **values)  # keep the chosen level if the mutation touched it
        if genome != champion and all(genome != g for _, g in made):
            made.append((f"adaptive {label}", genome))
    return [(f"{lineage}g{generation:04d}c{start_index + i:02d}", origin, genome)
            for i, (origin, genome) in enumerate(made)]


def run_generation(state: dict, runner: Runner, out: Path, args) -> None:
    generation = state["generation"] + 1
    champion = state["champion"]
    train = [{"id": name, "kind": "builtin"} for name in args.train_field] + state["hall_of_fame"]
    holdout = [{"id": name, "kind": "builtin"} for name in args.holdout_field]
    champion_genome = Genome.from_dict(champion["genome"])
    lineage = state.get("lineage", "")
    offset = seed_offset(lineage)
    tabu = [t["direction"] for t in state.get("tabu", []) if t["origin"] == champion["id"]]
    # Strong-adaptation candidates start with the first generation not yet begun when
    # this feature arrived (a generation already partly played keeps its candidates,
    # because cached matches are keyed by candidate name).
    if "adaptive_from" not in state:
        started = any(k[0].startswith(f"{lineage}g{generation:04d}c") for k in runner.cache)
        state["adaptive_from"] = generation + 1 if started else generation
        save_state(out, state)  # fixed now, so a restart rebuilds the same candidates
    adaptive = ADAPTIVE_PER_GENERATION if generation >= state["adaptive_from"] else 0
    # Wider gene ranges likewise start with the first generation not yet begun.
    if "wide_from" not in state:
        started = any(k[0].startswith(f"{lineage}g{generation:04d}c") for k in runner.cache)
        state["wide_from"] = generation + 1 if started else generation
        save_state(out, state)
    use_gene_ranges(generation >= state["wide_from"])
    candidates = make_candidates(generation, champion_genome,
                                 recent_best(out, since=state.get("pool_since", 0)),
                                 args.candidates - adaptive, lineage, tabu)
    candidates += adaptive_candidates(generation, champion_genome, adaptive, lineage,
                                      start_index=len(candidates))
    if tabu:
        print(f"  steering away from {len(tabu)} abandoned direction(s) from {champion['id']}")
    specs = [genome_spec(cid, genome) for cid, _, genome in candidates]
    print(f"\n=== Generation {generation}: {len(specs)} candidates vs champion {champion['id']}, "
          f"training field {len(train)}, holdout field {len(holdout)} ===", flush=True)

    # 1. Screen (training field only)
    seed = offset + 20_000 + generation
    h2h_seeds = [offset + 20_000 + generation, offset + 30_000 + generation]
    tasks = [(champion, opp, seed, DEALS) for opp in train]
    for spec in specs:
        tasks += [(spec, opp, seed, DEALS) for opp in train]
        tasks += [(spec, champion, s, DEALS) for s in h2h_seeds]
    results = runner.run(tasks, "screening")
    champ_rows = field_rows(results, champion, train, [seed])
    base = field_summary(champ_rows)
    screened = []
    for (cid, origin, genome), spec in zip(candidates, specs):
        rows = field_rows(results, spec, train, [seed])
        f, c = field_summary(rows), compare(rows, champ_rows, train)
        h = head_to_head(results, spec, champion, h2h_seeds)
        too_slow = bool(base["ms"] and f["ms"] and f["ms"] > SPEED_LIMIT * base["ms"])
        score = c["diff"] + H2H_WEIGHT * h["bb"]
        screened.append({"id": cid, "generation": generation, "origin": origin,
                         "genome": genome.to_dict(), "score": round(score, 4),
                         "field_diff": round(c["diff"], 4), "better_share": round(c["better_share"], 3),
                         "field_points": f["points"], "h2h_bb": round(h["bb"], 4),
                         "ms": f["ms"], "too_slow": too_slow})
    screened.sort(key=lambda r: -r["score"])
    print(f"  champion vs training field: {base['points']}/{base['matches']} points, "
          f"{base['bb']:+.3f} bb/deal, {base['ms']:.2f} ms/decision")
    for row in screened[:5]:
        print(f"  {row['id']} ({row['origin']}): score {row['score']:+.3f} | field "
              f"{row['field_diff']:+.3f} bb/deal vs champion, better vs "
              f"{row['better_share']:.0%} of opponents, {row['field_points']} pts | h2h "
              f"{row['h2h_bb']:+.3f}{' | TOO SLOW' if row['too_slow'] else ''}")

    # 2. Confirm on fresh seeds
    finalists = [r for r in screened if r["score"] > 0 and not r["too_slow"]][:args.finalists]
    promoted = None
    if finalists:
        conf_h2h = [offset + 40_000 + 10 * generation + k for k in range(CONFIRM_H2H_SEEDS)]
        conf_seeds = [offset + 50_000 + 10 * generation + k for k in range(CONFIRM_FIELD_SEEDS)]
        everyone = train + holdout
        tasks = [(champion, opp, s, DEALS) for s in conf_seeds for opp in everyone]
        final_specs = {r["id"]: genome_spec(r["id"], Genome.from_dict(r["genome"])) for r in finalists}
        for spec in final_specs.values():
            tasks += [(spec, champion, s, DEALS) for s in conf_h2h]
            tasks += [(spec, opp, s, DEALS) for s in conf_seeds for opp in everyone]
        results = runner.run(tasks, "confirming")
        champ_train = field_rows(results, champion, train, conf_seeds)
        champ_hold = field_rows(results, champion, holdout, conf_seeds)
        ct, ch = field_summary(champ_train), field_summary(champ_hold)
        best = None
        for row in finalists:
            spec = final_specs[row["id"]]
            rows_t = field_rows(results, spec, train, conf_seeds)
            rows_h = field_rows(results, spec, holdout, conf_seeds)
            ft, fh = field_summary(rows_t), field_summary(rows_h)
            cmp_t, cmp_h = compare(rows_t, champ_train, train), compare(rows_h, champ_hold, holdout)
            h = head_to_head(results, spec, champion, conf_h2h)
            checks = {
                "field_significant": cmp_t["diff"] - Z_ONE_SIDED_95 * cmp_t["se"] > 0,
                "breadth": cmp_t["better_share"] >= BREADTH,
                "field_points": ft["points"] >= ct["points"],
                "holdout_not_worse": cmp_h["diff"] >= 0 and fh["points"] >= ch["points"] - 1,
                "h2h_not_losing": h["bb"] >= 0,
            }
            passed = all(checks.values())
            row["confirm"] = {
                "train_diff": round(cmp_t["diff"], 4), "train_se": round(cmp_t["se"], 4),
                "train_better_share": round(cmp_t["better_share"], 3),
                "train_points": ft["points"], "champion_train_points": ct["points"],
                "holdout_diff": round(cmp_h["diff"], 4),
                "holdout_better_share": round(cmp_h["better_share"], 3),
                "holdout_points": fh["points"], "champion_holdout_points": ch["points"],
                "h2h_bb": round(h["bb"], 4), "h2h_se": round(h["se"], 4),
                "h2h_verdicts": _short(h["verdicts"]),
                "checks": {k: bool(v) for k, v in checks.items()}, "passed": passed}
            failed = [k for k, v in checks.items() if not v]
            print(f"  confirm {row['id']}: training {cmp_t['diff']:+.3f} ± {cmp_t['se']:.3f} bb/deal, "
                  f"better vs {cmp_t['better']}/{cmp_t['opponents']}, {ft['points']} vs "
                  f"{ct['points']} pts | holdout {cmp_h['diff']:+.3f}, {fh['points']} vs "
                  f"{ch['points']} pts | h2h {h['bb']:+.3f} ({_short(h['verdicts'])}) -> "
                  f"{'PASS' if passed else 'fail: ' + ', '.join(failed)}", flush=True)
            if passed and (best is None or cmp_t["diff"] > best[1]):
                best = (row, cmp_t["diff"])
        if best:
            promoted = best[0]
    else:
        print("  no candidate scored above the champion; nothing to confirm")

    with (out / "candidates.jsonl").open("a") as handle:
        for row in screened:
            handle.write(json.dumps(row) + "\n")
    state["generation"] = generation
    state["candidates_tried"] = state.get("candidates_tried", 0) + len(specs)
    event = {"generation": generation, "champion": champion["id"], "tried": len(specs),
             "best_screen": screened[0]["id"], "best_score": screened[0]["score"]}
    if promoted:
        state["hall_of_fame"] = state["hall_of_fame"] + [champion]  # every past champion, kept
        state["champion"] = genome_spec(promoted["id"], Genome.from_dict(promoted["genome"]))
        event["new_champion"] = promoted["id"]
        event["confirm"] = promoted["confirm"]
    banner(generation, champion["id"], promoted["id"] if promoted else None,
           len(state["hall_of_fame"]))
    state["history"].append(event)
    save_state(out, state)


def banner(generation: int, old: str, new: str | None, past: int) -> None:
    """A very visible end-of-generation verdict."""
    if new:
        lines = [f"GENERATION {generation}: CHAMPION DETHRONED",
                 f"{old}  -->  {new}",
                 f"{new} is the new champion; {old} joins the field ({past} past champions)"]
        char = "#"
    else:
        lines = [f"GENERATION {generation}: CHAMPION NOT DETHRONED",
                 f"{old} remains champion ({past} past champions in the field)"]
        char = "="
    width = max(len(line) for line in lines) + 8
    print("\n" + char * width)
    for line in lines:
        print(f"{char * 2}  {line:<{width - 8}}  {char * 2}")
    print(char * width + "\n", flush=True)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line another process is still writing
    return rows


def known_bots(dirs: list[Path]) -> dict[str, tuple[str, dict]]:
    """Every bot ever seen, as {id: (group, spec)}: all built-in strategies, the
    bundle's practice bots, and for every lineage folder its champions (current
    and past) and every candidate it generated."""
    bots: dict[str, tuple[str, dict]] = {
        name: ("built-in", {"id": name, "kind": "builtin"}) for name in BUILTINS}
    for folder in dirs:
        if not (folder / "state.json").exists():
            continue
        state = json.loads((folder / "state.json").read_text())
        tag = state.get("lineage") or "A"
        for row in _read_jsonl(folder / "candidates.jsonl"):
            bots.setdefault(row["id"], (f"lineage {tag} candidates",
                                        genome_spec(row["id"], Genome.from_dict(row["genome"]))))
        for spec in state["hall_of_fame"]:
            if spec["kind"] == "genome":
                bots[spec["id"]] = (f"lineage {tag} past champions", spec)
        bots[state["champion"]["id"]] = (f"lineage {tag} champion", state["champion"])
    return bots


def gauntlet_many(runner: Runner, players: list[dict], bots: dict, seeds: list[int],
                  label: str) -> dict[str, dict]:
    """Each player plays one duplicate match against every other known bot on each seed.
    All the matches go to the worker pool together. Returns {player id: report}."""
    tasks = [(player, spec, s, DEALS) for player in players for s in seeds
             for bot_id, (_, spec) in bots.items() if bot_id != player["id"]]
    results = runner.run(tasks, label)
    return {player["id"]: _summarize(player, {(k[1], k[2]): r for k, r in results.items()
                                              if k[0] == player["id"]}, bots, seeds)
            for player in players}


def gauntlet(runner: Runner, champion: dict, bots: dict, seeds: list[int], label: str) -> dict:
    """The champion plays one duplicate match against every other known bot on each seed."""
    return gauntlet_many(runner, [champion], bots, seeds, label)[champion["id"]]


def _summarize(champion: dict, rows: dict, bots: dict, seeds: list[int]) -> dict:
    groups: dict[str, dict] = {}
    losses = []
    for (opp_id, s), r in rows.items():
        group = bots[opp_id][0] if opp_id in bots else "other"
        g = groups.setdefault(group, {"matches": 0, "points": 0.0, "W": 0, "C": 0, "L": 0, "bb": 0.0})
        g["matches"] += 1
        g["points"] += r["points"][0]
        g["bb"] += r["mean_bb"]
        g["W" if r["points"][0] == 1 else "C" if r["points"][0] == 0.5 else "L"] += 1
        if r["points"][0] == 0:
            losses.append((opp_id, s, r["mean_bb"]))
    total = {k: sum(g[k] for g in groups.values()) for k in ("matches", "points", "W", "C", "L", "bb")}
    total["bb"] /= max(1, total["matches"])
    for g in groups.values():
        g["bb"] /= g["matches"]
    return {"champion": champion["id"], "seeds": seeds, "opponents": len({k[0] for k in rows}),
            "total": total, "groups": groups, "losses": sorted(losses, key=lambda x: x[2]),
            "rows": rows}


def _gauntlet_lines(report: dict) -> list[str]:
    t = report["total"]
    lines = [f"{report['champion']} vs every known bot ({report['opponents']} opponents, "
             f"seeds {report['seeds']}): {t['points']:.1f}/{t['matches']} points "
             f"({t['points'] / t['matches']:.0%}), W{t['W']} C{t['C']} L{t['L']}, "
             f"{t['bb']:+.3f} bb/deal"]
    for name, g in sorted(report["groups"].items()):
        lines.append(f"  {name:<28} {g['points']:>6.1f}/{g['matches']:<5} W{g['W']} C{g['C']} "
                     f"L{g['L']}  {g['bb']:+.3f} bb/deal")
    if report["losses"]:
        lines.append("  lost to: " + ", ".join(f"{o} ({bb:+.2f})" for o, _, bb in report["losses"][:12])
                     + (" ..." if len(report["losses"]) > 12 else ""))
    else:
        lines.append("  lost to: nobody")
    return lines


def periodic_check(state: dict, runner: Runner, out: Path, peers: list[Path]) -> None:
    """The current champion AND every past champion of this lineage play every bot ever
    seen (both lineages) on the same fresh seed. If a past champion is decidedly better
    than the current one (same rules as a promotion), the lineage reverts to it and
    stops evolving in the direction it had taken from it."""
    generation = state["generation"]
    offset = seed_offset(state.get("lineage", ""))
    seed = offset + 70_000 + generation
    current = state["champion"]
    past = [s for s in state["hall_of_fame"] if s["kind"] == "genome"]
    every = known_bots([out] + peers)
    champions = {k for k, (group, _) in every.items() if "champion" in group}
    kept, dropped = distinct(runner, out, {k: spec for k, (_, spec) in every.items()}, champions)
    bots = {k: every[k] for k in kept}
    print(f"\n  Periodic check after generation {generation}: current champion and "
          f"{len(past)} past champion(s) vs {len(bots) - 1} distinct bots "
          f"({len(dropped)} near-duplicates of {len(every)} known bots left out)", flush=True)
    reports = gauntlet_many(runner, [current] + past, bots, [seed], "periodic check")
    cur = reports[current["id"]]
    lines = [f"PERIODIC CHECK, GENERATION {generation}"] + _gauntlet_lines(cur)
    history = _read_jsonl(out / "periodic_checks.jsonl")
    if history:
        p = history[-1]
        lines.append(f"  previous check (generation {p['generation']}, champion {p['champion']}): "
                     f"{p['total']['points'] / p['total']['matches']:.0%} of points vs "
                     f"{p['opponents']} opponents, {p['total']['bb']:+.3f} bb/deal")

    # Past champions under the same conditions, compared on the same opponents.
    lines.append("Past champions on the same deals (compared with the current champion):")
    challengers, past_summary = [], []
    for spec in past:
        rep = reports[spec["id"]]
        common = [k for k in rep["rows"] if k in cur["rows"] and k[0] not in (spec["id"], current["id"])]
        mine = {k: rep["rows"][k] for k in common}
        theirs = {k: cur["rows"][k] for k in common}
        c = compare(mine, theirs, [])
        points_mine = sum(r["points"][0] for r in mine.values())
        points_theirs = sum(r["points"][0] for r in theirs.values())
        decided = (c["diff"] - Z_ONE_SIDED_95 * c["se"] > 0 and c["better_share"] >= BREADTH
                   and points_mine >= points_theirs)
        past_summary.append({"id": spec["id"], "diff": round(c["diff"], 4), "se": round(c["se"], 4),
                             "better_share": round(c["better_share"], 3), "points": points_mine,
                             "current_points": points_theirs, "decidedly_better": decided})
        lines.append(f"  {spec['id']:<12} {points_mine:.1f} vs {points_theirs:.1f} pts, "
                     f"{c['diff']:+.3f} ± {c['se']:.3f} bb/deal, better vs {c['better_share']:.0%} "
                     f"-> {'DECIDEDLY BETTER' if decided else 'not better'}")
        if decided:
            challengers.append((c["diff"], spec))
    revert_to = None
    for _, spec in sorted(challengers, key=lambda x: -x[0]):
        h2h_seeds = [offset + 80_000 + 10 * generation + k for k in range(CONFIRM_H2H_SEEDS)]
        results = runner.run([(spec, current, s, DEALS) for s in h2h_seeds], "revert head to head")
        h = head_to_head(results, spec, current, h2h_seeds)
        ok = h["bb"] >= 0
        lines.append(f"  {spec['id']} vs {current['id']} head to head: {h['bb']:+.3f} bb/deal "
                     f"({_short(h['verdicts'])}) -> {'reverting' if ok else 'loses, no revert'}")
        if ok:
            revert_to = spec
            break

    record = {k: v for k, v in cur.items() if k != "rows"}
    record.update(generation=generation, past_champions=past_summary,
                  reverted_to=revert_to["id"] if revert_to else None)
    with (out / "periodic_checks.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    width = max(len(line) for line in lines) + 4
    print("\n" + "*" * width)
    for line in lines:
        print(f"* {line:<{width - 4}} *")
    print("*" * width + "\n", flush=True)

    if revert_to:
        abandoned = direction(Genome.from_dict(revert_to["genome"]), Genome.from_dict(current["genome"]))
        state["hall_of_fame"] = [s for s in state["hall_of_fame"] if s["id"] != revert_to["id"]] + [current]
        state["champion"] = revert_to
        state.setdefault("tabu", []).append({"origin": revert_to["id"], "abandoned": current["id"],
                                             "generation": generation, "direction": abandoned})
        state["pool_since"] = generation
        state["history"].append({"generation": generation, "champion": current["id"],
                                 "reverted_to": revert_to["id"]})
        save_state(out, state)
        lines = [f"GENERATION {generation}: CHAMPION REVERTED",
                 f"{current['id']}  -->  {revert_to['id']} (a past champion)",
                 f"{revert_to['id']} beat it decidedly on the same deals; {current['id']} joins the field.",
                 f"New candidates will avoid the direction taken from {revert_to['id']} to {current['id']}."]
        width = max(len(line) for line in lines) + 8
        print("!" * width)
        for line in lines:
            print(f"!!  {line:<{width - 8}}  !!")
        print("!" * width + "\n", flush=True)


def final_compare(runner: Runner, out: Path, peers: list[Path], seeds: list[int]) -> None:
    """Every lineage's champion vs every known bot on several seeds, and vs each other."""
    dirs = [d for d in [out] + peers if (d / "state.json").exists()]
    champions = [json.loads((d / "state.json").read_text())["champion"] for d in dirs]
    every = known_bots(dirs)
    always = {k for k, (group, _) in every.items() if "champion" in group}
    kept, dropped = distinct(runner, out, {k: spec for k, (_, spec) in every.items()}, always)
    bots = {k: every[k] for k in kept}
    lines = [f"FINAL COMPARISON: {len(champions)} champion(s) vs {len(bots) - 1} distinct bots "
             f"({len(dropped)} near-duplicates of {len(every)} known bots left out), "
             f"seeds {seeds}, {DEALS} deals per match"]
    summary = []
    for champion in champions:
        report = gauntlet(runner, champion, bots, seeds, f"final {champion['id']}")
        lines += _gauntlet_lines(report)
        summary.append((report["total"]["points"] / report["total"]["matches"],
                        report["total"]["bb"], champion["id"]))
    if len(champions) == 2:
        a, b = champions
        h2h_seeds = [s + 500 for s in seeds] + [s + 600 for s in seeds]
        results = runner.run([(a, b, s, DEALS) for s in h2h_seeds], "champion vs champion")
        rows = list(results.values())
        lines.append(f"{a['id']} vs {b['id']} head to head over {len(rows)} matches: "
                     f"{st.mean(r['mean_bb'] for r in rows):+.3f} bb/deal for {a['id']} "
                     f"({_short([r['verdict'] for r in rows])})")
    summary.sort(reverse=True)
    lines.append("Ranking by share of points, then bb/deal: "
                 + "  >  ".join(f"{cid} ({share:.1%}, {bb:+.3f})" for share, bb, cid in summary))
    report = "\n".join(lines)
    print("\n" + report)
    (out / "final_comparison.txt").write_text(report + "\n")


def print_status(state: dict) -> None:
    print(f"Generations: {state['generation']}, candidates tried: {state.get('candidates_tried', 0)}")
    print(f"Champion: {state['champion']['id']}")
    print(f"Past champions (all kept in the training field): "
          f"{[s['id'] for s in state['hall_of_fame']]}")
    for event in state["history"]:
        if "new_champion" in event:
            print(f"  generation {event['generation']}: {event['new_champion']} "
                  f"replaced {event['champion']}")
        if "reverted_to" in event:
            print(f"  generation {event['generation']}: REVERTED from {event['champion']} "
                  f"to past champion {event['reverted_to']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="evolution")
    parser.add_argument("--generations", type=int, default=10_000)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--finalists", type=int, default=5,
                        help="How many of the best screened candidates are confirmed on fresh deals.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--lineage", default="",
                        help="Name for a separate search line (prefixes its bot names and "
                             "shifts its seeds). Only used when the folder is new.")
    parser.add_argument("--start", choices=("potshove08", "random"), default="potshove08",
                        help="First champion of a new lineage. Only used when the folder is new.")
    parser.add_argument("--peer", action="append", default=[], type=Path,
                        help="Another lineage's folder: its champions and candidates join the "
                             "periodic check and the final comparison. Repeatable.")
    parser.add_argument("--check-every", type=int, default=5,
                        help="Run the periodic check every N generations (0 = never).")
    parser.add_argument("--final", action="store_true",
                        help="Instead of searching: every lineage's champion vs every known bot, "
                             "and vs each other.")
    parser.add_argument("--final-seeds", default="90001,90002")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = load_state(out, args.lineage, args.start)
    if args.status:
        print_status(state)
        return
    evaluator.warmup()
    runner = Runner(out, args.workers)
    # Fields without near-duplicates: holdout bots that duplicate a training bot are dropped too.
    field_specs = {n: {"id": n, "kind": "builtin"} for n in TRAIN_FIELD + HOLDOUT_FIELD}
    kept, dropped = distinct(runner, out, field_specs)
    args.train_field = [n for n in TRAIN_FIELD if n in kept]
    args.holdout_field = [n for n in HOLDOUT_FIELD if n in kept]
    if dropped:
        print("Near-duplicates left out of the fields: "
              + ", ".join(f"{d} (plays like {twin})" for d, twin in dropped.items()))
    if args.final:
        try:
            final_compare(runner, out, args.peer, [int(s) for s in args.final_seeds.split(",")])
        finally:
            runner.close()
        return
    try:
        # A periodic check that a restart interrupted is redone first (its matches are cached).
        done = {r["generation"] for r in _read_jsonl(out / "periodic_checks.jsonl")}
        g = state["generation"]
        if args.check_every and g and g % args.check_every == 0 and g not in done:
            periodic_check(state, runner, out, args.peer)
        for _ in range(args.generations):
            run_generation(state, runner, out, args)
            if args.check_every and state["generation"] % args.check_every == 0:
                periodic_check(state, runner, out, args.peer)
    except KeyboardInterrupt:
        print("\nStopped; rerun the same command to resume.")
    finally:
        runner.close()
    print_status(state)


if __name__ == "__main__":
    main()
