import argparse
from pathlib import Path

from slot_forecasting.advanced import AllTicksForecaster  # noqa: F401  (kept for the plain adaptive_lags model)
from slot_forecasting.advanced2 import EpisodeShiftForecaster, GatedShrink
from slot_forecasting.io import load_inputs, read_labels, write_submission


def build_model():
    """Tuning-best episode shift (self cohort) plus the jackpot/balanced short-history shrink.

    Same model as the combo_shift_self_g0.7_s80_shrink_jb_0.5 candidate in run_candidates.py.
    """
    shift = EpisodeShiftForecaster(
        "self",
        half_life=32.0,
        shrink=80.0,
        gain=0.7,
    )
    return GatedShrink(
        shift,
        threshold=128,
        weight=0.5,
        machine_types=("jackpot", "balanced"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test",
                        help="Split to predict. The reproduce command uses the default, test.")
    args = parser.parse_args()

    train_queries, train_history = load_inputs(args.data / "train")
    labels = read_labels(args.data / "train" / "labels.csv", train_queries)
    queries, history = load_inputs(args.data / args.split)
    model = build_model()
    model.fit(train_queries, train_history, labels)
    write_submission(args.out, model.predict(queries, history))


if __name__ == "__main__":
    main()
