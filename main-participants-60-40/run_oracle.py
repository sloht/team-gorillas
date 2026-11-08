
"""
Oracle and error-structure diagnostics for the validation split.

Usage (from this folder):
    python run_oracle.py --data public --out oracle-01

DIAGNOSTIC ONLY. The oracle rows use the validation targets (true probability
vectors), so they are ceilings, not models. Never select a model with them.

Only train/ and validation/ are read; the test split is never loaded. Scoring
uses the project's subset_score on the full validation set. The output folder
must not already exist.
"""

import argparse
import hashlib
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_candidates import BASELINE, baseline_candidates
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import subset_score
from slot_forecasting.io import (
    load_inputs,
    read_labels,
    read_rows,
    read_submission,
    read_targets,
    require_ids,
    write_json,
)
from slot_forecasting.rounds import round_metadata
from slot_forecasting.schema import CONDITIONS, PAYOUTS, ValidationError
from slot_forecasting.scoring import distribution_error

N = len(PAYOUTS)
BLEND_STEPS = 20
GOAL_POINTS = 40.0
STARTER_METHODS = ("pooled", "recent", "pooled_recent")

HISTORY_BUCKETS = (
    (0, 1, "0"),
    (1, 8, "1-7"),
    (8, 32, "8-31"),
    (32, 128, "32-127"),
    (128, None, "128+"),
)

INPUT_FILES = [
    "metadata.json",
    "train/queries.csv",
    "train/history.csv",
    "train/labels.csv",
    "validation/queries.csv",
    "validation/history.csv",
    "validation/targets.csv",
    "validation/conditions.csv",
    "validation/reference.csv",
]


def mix(first, second, weight):
    return tuple(
        weight * a + (1 - weight) * b
        for a, b in zip(first, second)
    )


def weighted_error(prediction, targets):
    """Mean over episodes of the mean per-query squared error, like the contest scorer."""
    by_episode = defaultdict(list)

    for target in targets:
        by_episode[target.episode_id].append(
            distribution_error(
                prediction[target.query_id],
                target.probabilities,
            )
        )

    return (
        math.fsum(math.fsum(v) / len(v) for v in by_episode.values())
        / len(by_episode)
    )


def group_mean(targets, key_of):
    """Replace each query's forecast by the mean true distribution of its group."""
    sums = {}

    for target in targets:
        entry = sums.setdefault(key_of(target), [[0.0] * N, 0])

        for index, value in enumerate(target.probabilities):
            entry[0][index] += value

        entry[1] += 1

    result = {}

    for target in targets:
        values, count = sums[key_of(target)]

        result[target.query_id] = tuple(
            value / count for value in values
        )

    return result, len(sums)


def shift_corrected(model, targets, key_of):
    """Model plus the exact mean residual of its group, clipped at zero and renormalized."""
    sums = {}

    for target in targets:
        entry = sums.setdefault(key_of(target), [[0.0] * N, 0])

        for index, (truth, guess) in enumerate(
            zip(target.probabilities, model[target.query_id])
        ):
            entry[0][index] += truth - guess

        entry[1] += 1

    result = {}

    for target in targets:
        values, count = sums[key_of(target)]

        raw = [
            max(0.0, guess + shift / count)
            for guess, shift in zip(model[target.query_id], values)
        ]

        total = math.fsum(raw)

        result[target.query_id] = (
            tuple(value / total for value in raw)
            if total > 0
            else model[target.query_id]
        )

    return result, len(sums)


def best_blend(model, reference, targets, groups):
    """Per condition, the weight on the model (0..1) that minimizes validation error."""
    blended, weights = {}, {}

    for condition in CONDITIONS:
        subset = [
            target
            for target in targets
            if groups[target.query_id] == condition
        ]

        best = None

        for step in range(BLEND_STEPS + 1):
            weight = step / BLEND_STEPS

            candidate = {
                target.query_id: mix(
                    model[target.query_id],
                    reference[target.query_id],
                    weight,
                )
                for target in subset
            }

            error = weighted_error(candidate, subset)

            if best is None or error < best[0]:
                best = (error, weight, candidate)

        weights[condition] = best[1]
        blended.update(best[2])

    return blended, weights


def render(title, header, rows):
    table = [list(header)] + [
        [str(cell) for cell in row]
        for row in rows
    ]

    widths = [
        max(len(row[i]) for row in table)
        for i in range(len(header))
    ]

    lines = [title]

    for index, row in enumerate(table):
        lines.append(
            "  ".join(
                cell.ljust(widths[i]) if i == 0
                else cell.rjust(widths[i])
                for i, cell in enumerate(row)
            )
        )

        if index == 0:
            lines.append(
                "  ".join("-" * width for width in widths)
            )

    return "\n".join(lines)


def bucket_label(count):
    for low, high, label in HISTORY_BUCKETS:
        if count >= low and (high is None or count < high):
            return label

    return HISTORY_BUCKETS[-1][2]


def score_table(results):
    rows = []

    for name, result in results.items():
        rows.append([
            name,
            f"{result['technical_points']:.3f}",
            *[
                f"{result['by_condition'][c]['loss_ratio']:.3f}"
                for c in CONDITIONS
            ],
            f"{result['mean_loss_ratio']:.4f}",
        ])

    return render(
        "FULL validation (oracle rows are ceilings; diagnostic only)",
        ["row", "points", *CONDITIONS, "mean_ratio"],
        rows,
    )


def structure_tables(
    queries,
    groups,
    history_counts,
    model_error,
    reference_error,
):
    by_type = defaultdict(lambda: [0, 0.0, 0.0])
    by_bucket = defaultdict(lambda: [0, 0.0, 0.0])

    total_model = math.fsum(model_error.values())

    for query in queries:
        condition = groups[query.query_id]

        bucket = bucket_label(
            history_counts.get(
                (query.episode_id, query.machine_id),
                0,
            )
        )

        for stats in (
            by_type[(condition, query.machine_type)],
            by_bucket[(condition, bucket)],
        ):
            stats[0] += 1
            stats[1] += model_error[query.query_id]
            stats[2] += reference_error[query.query_id]

    order = {
        condition: index
        for index, condition in enumerate(CONDITIONS)
    }

    labels = [bucket[2] for bucket in HISTORY_BUCKETS]

    def ratio(stats):
        return stats[1] / max(stats[2], 1e-12)

    type_rows = [
        [
            condition,
            machine_type,
            stats[0],
            f"{stats[1] / stats[0]:.4f}",
            f"{stats[2] / stats[0]:.4f}",
            f"{ratio(stats):.3f}",
        ]
        for (condition, machine_type), stats in sorted(
            by_type.items(),
            key=lambda item: (
                order[item[0][0]],
                item[0][1],
            ),
        )
    ]

    bucket_rows = [
        [
            condition,
            bucket,
            stats[0],
            f"{stats[1] / stats[0]:.4f}",
            f"{stats[2] / stats[0]:.4f}",
            f"{ratio(stats):.3f}",
            f"{stats[1] / total_model:.3f}",
        ]
        for (condition, bucket), stats in sorted(
            by_bucket.items(),
            key=lambda item: (
                order[item[0][0]],
                labels.index(item[0][1]),
            ),
        )
    ]

    return (
        render(
            "Mean squared error per query by condition and machine_type (unweighted)",
            [
                "condition",
                "machine_type",
                "queries",
                "adaptive_lags",
                "reference",
                "ratio",
            ],
            type_rows,
        ),
        render(
            "Mean squared error per query by condition and history length (unweighted)",
            [
                "condition",
                "history",
                "queries",
                "adaptive_lags",
                "reference",
                "ratio",
                "share_of_error",
            ],
            bucket_rows,
        ),
    )


def target_note(max_points, current_ratio):
    if max_points < GOAL_POINTS:
        return (
            f"max_points={max_points:g}: "
            f"{GOAL_POINTS:g} points is impossible within this round alone."
        )

    needed = max_points / GOAL_POINTS - 1
    slope = max_points / (1 + current_ratio) ** 2

    return (
        f"{GOAL_POINTS:g} points at max_points={max_points:g} "
        f"needs a mean loss ratio <= {needed:.3f} "
        f"(baseline is {current_ratio:.3f}). Near the baseline, "
        f"0.01 of mean ratio is worth about {0.01 * slope:.2f} points."
    )


def run_oracle(data, out):
    public = Path(data).resolve()
    out = Path(out)

    if out.exists():
        raise ValidationError(
            "Choose a fresh output directory; existing folders are never overwritten."
        )

    metadata = round_metadata(public)
    max_points = metadata["max_points"]

    train_q, train_h = load_inputs(public / "train")
    val_q, val_h = load_inputs(public / "validation")

    labels = read_labels(
        public / "train/labels.csv",
        train_q,
    )

    targets = read_targets(
        public / "validation/targets.csv"
    )

    group_rows = read_rows(
        public / "validation/conditions.csv",
        ("query_id", "condition"),
    )

    groups = {
        row["query_id"]: row["condition"]
        for row in group_rows
    }

    if len(groups) != len(group_rows):
        raise ValidationError("Duplicate condition query ID.")

    reference = read_submission(
        public / "validation/reference.csv"
    )

    ids = {query.query_id for query in val_q}
    require_ids({target.query_id for target in targets}, ids)
    require_ids(groups, ids)
    require_ids(reference, ids)

    query_by_id = {
        query.query_id: query
        for query in val_q
    }

    episodes = {
        target.episode_id
        for target in targets
    }

    factories = (
        list(baseline_candidates())
        + [
            (method, lambda method=method: FrequencyForecaster(method))
            for method in STARTER_METHODS
        ]
    )

    predictions = {"reference": reference}

    for name, factory in factories:
        model = factory()

        fitted = model.fit(
            train_q,
            train_h,
            labels,
        )

        model = model if fitted is None else fitted

        prediction = model.predict(
            val_q,
            val_h,
        )

        require_ids(prediction, ids)
        predictions[name] = prediction

        print(f"fitted {name}", file=sys.stderr)

    baseline = predictions[BASELINE]

    def type_signal(target):
        query = query_by_id[target.query_id]
        return query.machine_type, query.signal

    def episode_type(target):
        return (
            target.episode_id,
            query_by_id[target.query_id].machine_type,
        )

    def episode_type_signal(target):
        query = query_by_id[target.query_id]

        return (
            target.episode_id,
            query.machine_type,
            query.signal,
        )

    group_counts = {}

    for name, key_of in (
        ("oracle_static_type_signal", type_signal),
        ("oracle_episode_mean", lambda target: target.episode_id),
        ("oracle_episode_type", episode_type),
        ("oracle_episode_type_signal", episode_type_signal),
    ):
        predictions[name], group_counts[name] = group_mean(
            targets,
            key_of,
        )

    (
        predictions["oracle_model_plus_episode_type_shift"],
        group_counts["oracle_model_plus_episode_type_shift"],
    ) = shift_corrected(
        baseline,
        targets,
        episode_type,
    )

    (
        predictions["oracle_blend_model_ref"],
        blend_weights,
    ) = best_blend(
        baseline,
        reference,
        targets,
        groups,
    )

    results = {
        name: subset_score(
            prediction,
            targets,
            groups,
            reference,
            episodes,
            max_points,
        )
        for name, prediction in predictions.items()
    }

    history_counts = Counter(
        (point.episode_id, point.machine_id)
        for point in val_h
    )

    model_error = {
        target.query_id: distribution_error(
            baseline[target.query_id],
            target.probabilities,
        )
        for target in targets
    }

    reference_error = {
        target.query_id: distribution_error(
            reference[target.query_id],
            target.probabilities,
        )
        for target in targets
    }

    type_table, bucket_table = structure_tables(
        val_q,
        groups,
        history_counts,
        model_error,
        reference_error,
    )

    baseline_result = results[BASELINE]

    notes = [
        target_note(
            max_points,
            baseline_result["mean_loss_ratio"],
        ),
        "Reference error per condition (the loss unit): "
        + ", ".join(
            f"{condition}="
            f"{baseline_result['by_condition'][condition]['reference_error']:.4f}"
            for condition in CONDITIONS
        ),
        "Queries per oracle group: "
        + ", ".join(
            f"{name}: {len(val_q)} queries / {count} groups"
            for name, count in group_counts.items()
        ),
        "oracle_blend_model_ref weights on adaptive_lags "
        "(rest on reference): "
        + ", ".join(
            f"{condition}={blend_weights[condition]:.2f}"
            for condition in CONDITIONS
        ),
    ]

    text = "\n\n".join([
        score_table(results),
        type_table,
        bucket_table,
        "\n".join(notes),
    ])

    out.mkdir(parents=True)

    (out / "table.txt").write_text(
        text + "\n",
        encoding="utf-8",
    )

    write_json(
        out / "oracle.json",
        {
            "round": metadata["round"],
            "season_id": metadata["season_id"],
            "max_points": max_points,
            "diagnostic_only": True,
            "note": (
                "Oracle rows use validation targets. "
                "They are ceilings and must not be used to select models."
            ),
            "results": results,
            "oracle_group_counts": group_counts,
            "blend_weights_on_model": blend_weights,
            "input_sha256": {
                name: hashlib.sha256(
                    (public / name).read_bytes()
                ).hexdigest()
                for name in INPUT_FILES
            },
        },
    )

    return text


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to a public/ folder.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output folder (must not exist).",
    )

    args = parser.parse_args(argv)

    print(run_oracle(args.data, args.out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
