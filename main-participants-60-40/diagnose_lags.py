
import argparse
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slot_forecasting.io import load_inputs  # noqa: E402


LAGS = tuple(range(1, 9))
TARGETS = ("paid", "big")
CROSS_REPEATS = 20

if hasattr(int, "bit_count"):
    popcount = int.bit_count
else:  # Python < 3.10
    def popcount(value):
        return bin(value).count("1")


class Machine:
    __slots__ = ("episode", "mtype", "bits", "vals", "obs", "regions")

    def __init__(self, episode, mtype, seq, full, even, odd):
        ticks = sorted(seq)

        self.episode = episode
        self.mtype = mtype
        self.bits = [1 << t for t in ticks]
        self.vals = [seq[t] for t in ticks]

        self.obs = 0
        for bit in self.bits:
            self.obs |= bit

        split_tick = ticks[len(ticks) // 2]
        first = ((1 << split_tick) - 1) & full

        self.regions = {
            "full": full,
            "odd": odd,
            "even": even,
            "first": first,
            "second": full & ~first,
        }


def build_machines(queries, history):
    type_of = {
        (q.episode_id, q.machine_id): q.machine_type
        for q in queries
    }

    seqs = defaultdict(dict)
    max_tick = 0

    for play in history:
        seqs[(play.episode_id, play.machine_id)][play.tick] = play.payout
        max_tick = max(max_tick, play.tick)

    width = max_tick + 1
    full = (1 << width) - 1
    even = sum(1 << t for t in range(0, width, 2))
    odd = full & ~even

    machines = []

    for key in sorted(seqs):
        machines.append(
            Machine(key[0], type_of[key], seqs[key], full, even, odd)
        )

    return machines


def masks_from(bits, vals):
    paid = big = 0

    for bit, value in zip(bits, vals):
        if value > 0:
            paid |= bit

            if value >= 2:
                big |= bit

    return paid, big


def lifts(obs, paid, big, lag, region, min_n):
    """Return ((L, null_var) for paid, (L, null_var) for big),
    or None if under-observed.
    """

    pairs = obs & (obs << lag) & region

    prev = paid << lag
    after_paid = pairs & prev
    after_unpaid = pairs & ~prev

    n1 = popcount(after_paid)
    n0 = popcount(after_unpaid)

    if n1 < min_n or n0 < min_n:
        return None

    out = []

    for target in (paid, big):
        h1 = popcount(after_paid & target)
        h0 = popcount(after_unpaid & target)

        pooled = (h1 + h0) / (n1 + n0)

        out.append((
            h1 / n1 - h0 / n0,
            pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n0)
        ))

    return tuple(out)


def split_lifts(machine, paid, big, lag, min_obs, min_half, names):
    full = lifts(
        machine.obs,
        paid,
        big,
        lag,
        machine.regions["full"],
        min_obs
    )

    if full is None:
        return None

    result = {"full": full}

    for name in names:
        result[name] = lifts(
            machine.obs,
            paid,
            big,
            lag,
            machine.regions[name],
            min_half
        )

    return result


def pearson(xs, ys):
    n = len(xs)

    if n < 3:
        return None

    mx = math.fsum(xs) / n
    my = math.fsum(ys) / n

    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)

    if sxx <= 1e-15 or syy <= 1e-15:
        return None

    sxy = math.fsum(
        (x - mx) * (y - my)
        for x, y in zip(xs, ys)
    )

    return sxy / math.sqrt(sxx * syy)


def reliability(records, ti, a, b):
    xs, ys = [], []

    for _, res in records:
        ra, rb = res[a], res[b]

        if ra is None or rb is None:
            continue

        xs.append(ra[ti][0])
        ys.append(rb[ti][0])

    return pearson(xs, ys)


def spread(records, ti):
    values = [
        res["full"][ti][0]
        for _, res in records
    ]

    variances = [
        res["full"][ti][1]
        for _, res in records
    ]

    if len(values) < 2:
        return None, None

    return (
        statistics.stdev(values),
        math.sqrt(math.fsum(variances) / len(variances))
    )


def sharing(records, ti, rng):
    """Return (r_within_episode, r_other_episodes) for same-type machines."""

    values = [
        res["full"][ti][0]
        for _, res in records
    ]

    by_episode = defaultdict(list)

    for i, (machine, _) in enumerate(records):
        by_episode[machine.episode].append(i)

    xs, ys, needs = [], [], []

    for i, (machine, _) in enumerate(records):
        others = [
            j for j in by_episode[machine.episode]
            if j != i
        ]

        if not others:
            continue

        xs.append(values[i])
        ys.append(
            math.fsum(values[j] for j in others) / len(others)
        )

        needs.append((i, len(others)))

    r_within = pearson(xs, ys)

    if r_within is None:
        return None, None

    total = len(records)
    cross_x, cross_plan = [], []

    for i, k in needs:
        available = total - len(
            by_episode[records[i][0].episode]
        )

        if available <= 0:
            continue

        cross_x.append(values[i])
        cross_plan.append((i, min(k, available)))

    if len(cross_x) < 3:
        return r_within, None

    rs = []

    for _ in range(CROSS_REPEATS):
        cross_y = []

        for i, k in cross_plan:
            episode = records[i][0].episode
            chosen = set()

            while len(chosen) < k:
                j = rng.randrange(total)

                if records[j][0].episode != episode:
                    chosen.add(j)

            cross_y.append(
                math.fsum(values[j] for j in chosen) / k
            )

        r = pearson(cross_x, cross_y)

        if r is not None:
            rs.append(r)

    return (
        r_within,
        statistics.fmean(rs) if rs else None
    )


def num(value, width=7, digits=3):
    if value is None:
        return "n/a".rjust(width)

    return f"{value:{width}.{digits}f}"


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose lag signal in the training history."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="public folder that contains train/"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345
    )

    parser.add_argument(
        "--min-obs",
        type=int,
        default=20,
        help="minimum pairs per condition for a machine/lag (default 20)"
    )

    parser.add_argument(
        "--min-half",
        type=int,
        default=10,
        help="minimum pairs per condition inside each split half (default 10)"
    )

    parser.add_argument(
        "--shuffles",
        type=int,
        default=20,
        help="null shuffles (default 20)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="reliability threshold for the final list (default 0.25)"
    )

    args = parser.parse_args()

    started = time.time()

    train_dir = Path(args.data) / "train"

    if not train_dir.is_dir():
        parser.error(f"{train_dir} is not a directory.")

    queries, history = load_inputs(train_dir)
    machines = build_machines(queries, history)

    rng = random.Random(args.seed)

    types = sorted({m.mtype for m in machines})

    print(
        f"Loaded {len(machines)} machines with history from {train_dir} "
        f"({', '.join(f'{t}={sum(1 for m in machines if m.mtype == t)}' for t in types)})."
    )

    print(
        f"min_obs={args.min_obs} min_half={args.min_half} "
        f"shuffles={args.shuffles} seed={args.seed}"
    )

    # Observed per-machine lifts.
    records = defaultdict(list)

    split_names = ("odd", "even", "first", "second")

    for machine in machines:
        paid, big = masks_from(machine.bits, machine.vals)

        for lag in LAGS:
            res = split_lifts(
                machine,
                paid,
                big,
                lag,
                args.min_obs,
                args.min_half,
                split_names
            )

            if res is not None:
                records[(machine.mtype, lag)].append((machine, res))

    # Null: shuffle payouts across each machine's observed ticks,
    # odd vs even reliability.
    null = defaultdict(lambda: ([], []))

    for machine in machines:
        for s in range(args.shuffles):
            vals = list(machine.vals)
            rng.shuffle(vals)

            paid, big = masks_from(machine.bits, vals)

            for lag in LAGS:
                res = split_lifts(
                    machine,
                    paid,
                    big,
                    lag,
                    args.min_obs,
                    args.min_half,
                    ("odd", "even")
                )

                if (
                    res is None
                    or res["odd"] is None
                    or res["even"] is None
                ):
                    continue

                for ti in range(len(TARGETS)):
                    xs, ys = null[
                        (machine.mtype, lag, ti, s)
                    ]

                    xs.append(res["odd"][ti][0])
                    ys.append(res["even"][ti][0])

    summary = {}

    header = (
        f"{'type':<9}{'lag':>3}{'nM':>5} | "
        f"{'rel_oe':>7}{'rel_fs':>7} | "
        f"{'null_mu':>7}{'null_sd':>8} | "
        f"{'sd_obs':>7}{'sd_bin':>7}{'ratio':>7} | "
        f"{'r_in':>7}{'r_out':>7}"
    )

    for ti, target in enumerate(TARGETS):
        print(
            f"\n=== target: {target} "
            f"({'payout > 0' if target == 'paid' else 'payout >= 2'}) ==="
        )

        print(header)
        print("-" * len(header))

        for mtype in types:
            for lag in LAGS:
                recs = records.get((mtype, lag), [])

                if not recs:
                    print(
                        f"{mtype:<9}{lag:>3}{0:>5} | "
                        "(no machine meets the observation minimums)"
                    )
                    continue

                rel_oe = reliability(
                    recs, ti, "odd", "even"
                )

                rel_fs = reliability(
                    recs, ti, "first", "second"
                )

                null_rs = [
                    r
                    for r in (
                        pearson(*null[(mtype, lag, ti, s)])
                        for s in range(args.shuffles)
                    )
                    if r is not None
                ]

                null_mu = (
                    statistics.fmean(null_rs)
                    if null_rs else None
                )

                null_sd = (
                    statistics.stdev(null_rs)
                    if len(null_rs) > 1 else None
                )

                sd_obs, sd_bin = spread(recs, ti)

                ratio = (
                    sd_obs / sd_bin
                    if sd_obs is not None and sd_bin
                    else None
                )

                r_in, r_out = sharing(recs, ti, rng)

                summary[(mtype, lag, ti)] = (
                    rel_oe,
                    rel_fs
                )

                print(
                    f"{mtype:<9}{lag:>3}{len(recs):>5} | "
                    f"{num(rel_oe)}{num(rel_fs)} | "
                    f"{num(null_mu)}{num(null_sd, 8)} | "
                    f"{num(sd_obs)}{num(sd_bin)}{num(ratio)} | "
                    f"{num(r_in)}{num(r_out)}"
                )

    print(
        f"\n=== (type, lag) with split-half reliability > "
        f"{args.threshold} ==="
    )

    flagged = {}

    for (mtype, lag, ti), (rel_oe, rel_fs) in sorted(summary.items()):
        if (
            (rel_oe is not None and rel_oe > args.threshold)
            or
            (rel_fs is not None and rel_fs > args.threshold)
        ):
            flagged.setdefault((mtype, lag), []).append(
                (TARGETS[ti], rel_oe, rel_fs)
            )

    if not flagged:
        print("(none)")

    for (mtype, lag), details in sorted(flagged.items()):
        text = "; ".join(
            f"{name}: odd/even={num(oe, 1, 3).strip()} "
            f"first/second={num(fs, 1, 3).strip()}"
            for name, oe, fs in details
        )

        print(f"({mtype}, lag {lag})  {text}")

    print(
        "\nNote: entries are listed when either split-half reliability "
        "exceeds the threshold; a high odd/even value with a low "
        "first/second value points to drift."
    )

    print(f"Elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
