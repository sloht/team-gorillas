# House Edge — submission README template

Copy this into your submission as `README.md`. Replace the bracketed fields with your actual details. Keep it short; this is a setup guide, not a graded writeup.

## Entry

- Team ID: [team ID]
- Round: [main or adaptation]
- Season ID: [copy `season_id` from this round's `public/metadata.json`]
- Forecast file: [path to your submitted CSV]
- Forecast SHA-256: [printed by `check_submission.py`]
- Code version: [commit ID or archive/version name]
- Environment: [`python` in `reproduce.json`; list any extra packages below]

## Setup

- Python/runtime version: [version]
- OS tested: [OS]
- Dependencies: [`requirements.txt` with exact versions; or "Python standard library only"]
- Required hardware: [CPU/RAM; GPU and version if used]
- Put the released data at: [path, for example `public/`]

The organizer rerun mounts the current round at `/data`, your code at `/submission` (read-only), and writable scratch/output at `/tmp` and `/output`. Adaptation also mounts main-round public data at `/main_data`. Your `reproduce.json` command must write `/output/submission.csv` without network access. Include this README and `reproduce.json` at the code ZIP root. Organizers prepare the pinned dependencies in an isolated environment before running predictions offline.

## Recreate the forecast

Run from: [folder]

```sh
[exact command that recreates the submitted test CSV]
```

- Output file: [path]
- Required model/config files: [paths; say "none" if the command trains everything]
- Random seeds and settings: [values or config path]
- What the command does: [loads a saved model / trains and predicts]
- Measured runtime and hardware: [time and machine used]
- Training command, if separate: [exact command and required inputs]
- Main-round data/model reused in adaptation, if any: [what and where]

If reruns are not deterministic, describe the source of variation and how the submitted predictions were produced. Include the saved checkpoint/settings needed to reproduce them.

## Local check

```sh
python check_submission.py [path/to/submission.csv] --data [path/to/current-round/public]
```

- Checker result: [valid / unresolved issue]
- Fresh-folder reproduction checked: [yes/no, plus any limitations]

Local validation does not submit an entry. Upload the forecast CSV (up to 10 MB) and code ZIP (up to 100 MB) through the correct round's Google Form, click Submit, and wait for confirmation. Keep the exact files. The latest response before cutoff counts, even if invalid; organizers validate files and check prize contenders' code afterward. The rerun budget is 30 minutes, two CPU cores, and 16 GiB RAM.
