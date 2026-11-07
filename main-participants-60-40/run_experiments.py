"""Edit starter_candidates() or supply your own named model factories."""
import argparse
from pathlib import Path
from slot_forecasting.experiments import run_experiments

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "public"))
parser.add_argument("--out", default="experiments")
args = parser.parse_args()
report = run_experiments(args.data, args.out)
for name, row in report["comparison"].items():
    print(f"{name}: tuning points {row['tuning']['technical_points']:.3f}")
print("Selected:", report["chosen"])
print("Fresh audit episode points:", round(report["audit"]["technical_points"], 3))
print("Submission:", Path(args.out) / "submission.csv")
