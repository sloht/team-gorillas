"""Score named candidate models on the tuning half, the audit half, and full validation.

Usage (from this folder):
    python run_candidates.py --data public --out candidates-01

Rules enforced here:
  * Only train/ and validation/ are read. The test split is never loaded.
  * Folds come from slot_forecasting.experiments.episode_folds().
  * Pick candidates by the tuning half only. The audit half is reported for
    information; do not select on it, and do not grid-search against it.
  * The output folder must not already exist.

Extra candidates are picked up from slot_forecasting/advanced2.py when it defines
candidates() returning a list of (name, zero-argument model factory) pairs.
"""
import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slot_forecasting.advanced import AllTicksForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import (load_inputs, read_labels, read_rows, read_submission,
                                 read_targets, require_ids, write_json, write_submission)
from slot_forecasting.rounds import round_metadata
from slot_forecasting.schema import CONDITIONS, ValidationError

BASELINE = "adaptive_lags"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
POINT_KEYS = ("technical_points", "points", "technical_score", "total_points", "score")
RATIO_KEYS = ("mean_loss_ratio", "loss_ratio", "ratio")
TIME_WARNING_SECONDS = 300.0


def baseline_candidates():
    return [(BASELINE, lambda: AllTicksForecaster(
        "lags", strength=24, local_weight=0.40, half_lives=(8, 24, 100), adaptive=True))]


def extra_candidates():
    try:
        module = importlib.import_module("slot_forecasting.advanced2")
    except ModuleNotFoundError as error:
        if error.name != "slot_forecasting.advanced2":
            raise
        return []
    if not hasattr(module, "candidates"):
        return []
    return list(module.candidates())


def all_candidates(only=None):
    candidates = baseline_candidates() + extra_candidates()
    names = [name for name, _ in candidates]
    if len(set(names)) != len(names):
        raise ValidationError("Candidate names must be unique.")
    for name in names:
        if not NAME_PATTERN.match(name):
            raise ValidationError(f"Candidate name {name!r} may only use letters, digits, _ . -")
    if only:
        wanted = [n for n in only if n]
        unknown = [n for n in wanted if n not in names]
        if unknown:
            raise ValidationError(f"Unknown candidates: {', '.join(unknown)}")
        candidates = [(n, f) for n, f in candidates if n == BASELINE or n in wanted]
    return candidates


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def extract_points(result):
    for key in POINT_KEYS:
        value = result.get(key)
        if _number(value):
            return value
        if isinstance(value, dict):
            for sub in ("technical", "total", "points"):
                if _number(value.get(sub)):
                    return value[sub]
    return None


def extract_condition_ratio(result, condition):
    """Find the loss ratio for one condition without assuming the exact result layout."""
    hits = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == condition:
                    hits.append((path + (str(key),), value))
                walk(value, path + (str(key),))
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and item.get("condition") == condition:
                    hits.append((path + (condition,), item))
                walk(item, path)

    walk(result, ())
    best = None
    for path, value in hits:
        if isinstance(value, dict):
            for key in RATIO_KEYS:
                if _number(value.get(key)):
                    if best is None or best[0] > 0:
                        best = (0, value[key])
                    break
        elif _number(value) and any("ratio" in part for part in path[:-1]):
            if best is None:
                best = (1, value)
    return None if best is None else best[1]


def summarize(result):
    return {
        "points": extract_points(result),
        "ratios": {c: extract_condition_ratio(result, c) for c in CONDITIONS},
        "mean_loss_ratio": result.get("mean_loss_ratio"),
    }


def _fmt(value, places=3):
    return f"{value:.{places}f}" if _number(value) else "n/a"


def format_table(title, names, summaries, seconds):
    header = ["candidate", "points", *CONDITIONS, "mean_ratio", "seconds"]
    rows = [header]
    for name in names:
        s = summaries[name]
        rows.append([name, _fmt(s["points"]), *[_fmt(s["ratios"][c]) for c in CONDITIONS],
                     _fmt(s["mean_loss_ratio"], 4), _fmt(seconds[name], 1)])
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [title]
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[0]) if i == 0 else cell.rjust(widths[i])
                               for i, cell in enumerate(row)))
        if index == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_candidates(data, out, only=None, skip_audit=False):
    public, out = Path(data).resolve(), Path(out)
    if out.exists():
        raise ValidationError("Choose a fresh output directory; existing folders are never overwritten.")
    metadata = round_metadata(public)
    candidates = all_candidates(only)

    train_q, train_h = load_inputs(public / "train")
    val_q, val_h = load_inputs(public / "validation")
    train = SimpleNamespace(queries=train_q, history=train_h,
                            labels=read_labels(public / "train/labels.csv", train_q))
    targets = read_targets(public / "validation/targets.csv")
    group_rows = read_rows(public / "validation/conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    if len(groups) != len(group_rows):
        raise ValidationError("Duplicate condition query ID.")
    reference = read_submission(public / "validation/reference.csv")
    ids = {q.query_id for q in val_q}
    require_ids({t.query_id for t in targets}, ids)
    require_ids(groups, ids)
    require_ids(reference, ids)

    tuning, audit = episode_folds(targets, groups)
    full = {t.episode_id for t in targets}
    splits = {"tuning": tuning, "full": full}
    if not skip_audit:
        splits["audit"] = audit
    max_points = metadata["max_points"]

    raw, summaries, seconds, predictions = {}, {}, {}, {}
    for name, factory in candidates:
        start = perf_counter()
        model = factory()
        fitted = model.fit(train.queries, train.history, train.labels)
        model = model if fitted is None else fitted
        prediction = model.predict(val_q, val_h)
        require_ids(prediction, ids)
        raw[name] = {split: subset_score(prediction, targets, groups, reference, episodes, max_points)
                     for split, episodes in splits.items()}
        seconds[name] = perf_counter() - start
        summaries[name] = {split: summarize(result) for split, result in raw[name].items()}
        predictions[name] = prediction
        print(f"finished {name} in {seconds[name]:.1f}s", file=sys.stderr)
        if seconds[name] > TIME_WARNING_SECONDS:
            print(f"WARNING: {name} took longer than {TIME_WARNING_SECONDS:.0f}s for fit + validation predict.",
                  file=sys.stderr)

    names = [name for name, _ in candidates]
    tables = []
    for split in ("tuning", "audit", "full"):
        if split in splits:
            label = {"tuning": "TUNING half (select on this)",
                     "audit": "AUDIT half (report only; do not select on this)",
                     "full": "FULL validation"}[split]
            tables.append(format_table(label, names, {n: summaries[n][split] for n in names}, seconds))
    chosen = min(names, key=lambda n: (raw[n]["tuning"]["mean_loss_ratio"], n))
    tables.append(f"Best by tuning mean loss ratio: {chosen}")
    text = "\n\n".join(tables)

    missing = any(summaries[n][s]["points"] is None or any(v is None for v in summaries[n][s]["ratios"].values())
                  for n in names for s in summaries[n])
    if missing:
        probe = json.dumps(raw[names[0]]["full"], sort_keys=True, default=str)[:1500]
        text += ("\n\nNOTE: some columns could not be located in the contest_score result. "
                 "Raw result for the first candidate (truncated):\n" + probe)

    out.mkdir(parents=True)
    (out / "predictions").mkdir()
    for name in names:
        write_submission(out / "predictions" / f"{name}.csv", predictions[name])
    (out / "table.txt").write_text(text + "\n", encoding="utf-8")
    input_files = ["metadata.json", "train/queries.csv", "train/history.csv", "train/labels.csv",
                   "validation/queries.csv", "validation/history.csv", "validation/targets.csv",
                   "validation/conditions.csv", "validation/reference.csv"]
    write_json(out / "candidates.json", {
        "round": metadata["round"], "season_id": metadata["season_id"], "max_points": max_points,
        "selection_rule": "Lowest tuning mean_loss_ratio. Audit is reported only.",
        "chosen_by_tuning": chosen,
        "tuning_episodes": sorted(tuning), "audit_episodes": sorted(audit),
        "audit_scored": not skip_audit,
        "seconds": seconds, "summaries": summaries, "raw": raw,
        "input_sha256": {name: sha256(public / name) for name in input_files},
        "source_sha256": {name: sha256(Path(__file__).resolve().parent / "slot_forecasting" / name)
                          for name in ("advanced.py", "advanced2.py")
                          if (Path(__file__).resolve().parent / "slot_forecasting" / name).exists()},
    })
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="Path to a public/ folder.")
    parser.add_argument("--out", type=Path, help="New output folder (must not exist).")
    parser.add_argument("--only", default="", help="Comma-separated candidate names (baseline is always run).")
    parser.add_argument("--skip-audit", action="store_true", help="Do not score the audit half.")
    parser.add_argument("--list", action="store_true", help="List candidate names and exit.")
    args = parser.parse_args(argv)
    if args.list:
        for name, _ in all_candidates():
            print(name)
        return 0
    if args.data is None or args.out is None:
        parser.error("--data and --out are required.")
    only = [n.strip() for n in args.only.split(",")] if args.only else None
    print(run_candidates(args.data, args.out, only=only, skip_audit=args.skip_audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
