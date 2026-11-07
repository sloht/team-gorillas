import argparse
from pathlib import Path

from slot_forecasting.advanced import AllTicksForecaster
from slot_forecasting.io import load_inputs, read_labels, write_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_queries, train_history = load_inputs(args.data / "train")
    labels = read_labels(args.data / "train" / "labels.csv", train_queries)
    test_queries, test_history = load_inputs(args.data / "test")
    model = AllTicksForecaster(
        "lags",
        strength=24,
        local_weight=0.40,
        half_lives=(8, 24, 100),
        adaptive=True,
    )
    model.fit(train_queries, train_history, labels)
    write_submission(args.out, model.predict(test_queries, test_history))


if __name__ == "__main__":
    main()
