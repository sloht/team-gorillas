# House Edge — Slot Machine Forecasting

This repository branch contains work for the TAMU Datathon 2026 House Edge
challenge. The task is to predict the next payout distribution for each query:
0x, 1x, 2x, 5x, or 20x. The score compares predicted probabilities with the
simulator's true payout probabilities, not just the single payout that occurred.

The participant kit, data, starter scripts, scoring code, and detailed
submission instructions are in [`main-participants-60-40/`](./main-participants-60-40/).
The main round is worth 60 points; adaptation is worth 40. The four conditions
(stable, short history, drift, and shared change) have equal weight in the
normalized squared-error score. See the kit's [scoring guide](./main-participants-60-40/README.md#4-scoring).

## Current status

- Current branch: `house-edge-baseline`.
- The baseline forecast and its reproduction archive are tracked:
  - [`outputs/submission.csv`](./main-participants-60-40/outputs/submission.csv)
  - [`entry.zip`](./main-participants-60-40/entry.zip)
- A `pooled_recent` forecast and matching archive are also tracked:
  - [`pooled_recent/submission.csv`](./main-participants-60-40/pooled_recent/submission.csv)
  - [`pooled_recent-entry.zip`](./main-participants-60-40/pooled_recent-entry.zip)
- On released validation data, `pooled_recent` scored 30.826/60 versus 30.000
  for the pooled reference. The test outcomes are private, so no test score is
  available locally.
- An experimental `adaptive_lags` model scored 31.094/60 on full validation
  and 30.919/60 on the held-out audit episodes, versus 30.826 and 30.565 for
  `pooled_recent`, respectively. Its validation error was lower than the
  reference in all four conditions. This is a modest validation improvement,
  not a guarantee of better hidden-test performance.
- The adaptive model's forecast and reproduction archive passed local format
  checks, and its packaged code reproduced its test forecast exactly. These
  experimental files are currently local and untracked; they have not been
  pushed or submitted to the contest.
- No contest form submission has been made.

## Reproduce and evaluate

Run commands from the kit directory:

```sh
cd main-participants-60-40
python3 run_starter.py --out outputs
python3 check_submission.py outputs/submission.csv
python3 check_submission.py outputs/validation_predictions.csv --split validation
python3 build_entry.py --starter outputs --out entry.zip
```

For the advanced experiment, the untracked source currently lives in
`main-participants-60-40/slot_forecasting/advanced.py`; the selected test
forecast and experiment report are in `advanced-experiments-01/`, and the
reproduction source/archive are in `advanced-entry/` and
`adaptive-lags-entry.zip`. Its predictor is pure Python and uses no third-party
packages. See the kit's [reproduction requirements](./main-participants-60-40/README.md#package-a-custom-model)
before packaging or submitting.

## Suggested continuation

1. Review the advanced model source, its report, and the exact forecast/code
   archive before deciding whether to use it.
2. Keep the existing tracked `pooled_recent` files as a fallback; do not
   replace them without comparing a candidate on validation and audit episodes.
3. If continuing model work, test changes on tuning episodes and report results
   on the separate audit episodes, including each condition.
4. Before any contest upload, check the exact CSV and matching ZIP, and obtain
   the Main form link. No upload has been made yet.
