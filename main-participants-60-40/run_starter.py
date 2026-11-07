"""Participant-only entry point: no organizer modules or private data required."""
import argparse
from pathlib import Path
from slot_forecasting.starter import run_starter

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "public"))
parser.add_argument("--out", default="outputs")
parser.add_argument("--method", default="pooled")
args = parser.parse_args()
report = run_starter(args.data, args.out, method=args.method)
for name, row in report["comparison"].items():
    print(f"{name}: validation technical points {row['contest']['technical_points']:.3f} / {row['contest']['max_points']:g}")
print("Submission:", Path(args.out) / "submission.csv")
