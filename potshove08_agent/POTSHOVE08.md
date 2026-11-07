# PotShove08 tournament submission

This folder is a complete copy of the official
[heads-up-bundle](https://github.com/tamu-datathon-org/heads-up-bundle) at
commit `c83ab6e`, with `agent.py` replaced by **PotShove08**, the top strategy
in the team's qualifying round robin. It runs on its own: it needs nothing from
the rest of this repo.

## What changed from the bundle

| File | Change |
|---|---|
| `agent.py` | `PARTICIPANT` and `AGENT_NAME` set; the three functions play PotShove08. Server plumbing untouched. |
| `test_agent.py` | Two tests added at the bottom for the time safety valve. |
| Everything else | Unchanged copies (`game.py`, `evaluator.py`, `judge_engine.py` match `../official/`). |

## How it plays

The same decisions as `poker_bots.PotShove08` played through
`official_match.OfficialSeat`: a 14-sample equity estimate against one random
hand, compared with thresholds adjusted for pot odds, position, street and
board texture; raises always aim for the pot-sized raise; a 10% bluff when
checked to with enough equity.

**Safety valve:** after 5 s of thinking in a half it uses 6 equity samples,
and after 6.5 s it uses 2, so it can't run out of the 8 s time bank (which
would leave it only checking or folding for the rest of the half). A half
normally uses 2-4 s, so in normal play this never triggers.

## Checked

- **Same decisions as the original:** 67,133 decisions over 20 full halves (10
  opponents, both seats), 0 mismatches, every action legal.
- **Bundle tests:** `python -m pytest -q` passes, 14 tests.
- **Real judge match** against `baseline_calling_station.py`, each half on a
  fresh server: won the match; 2.3 s of 8 s thinking per half; 0 errors.
- **Not checked:** the Docker build (`./tools/check_build.sh` needs Docker,
  which this machine doesn't have), and speed on the tournament's CPU.

## Submitting

The tournament builds `agent.py` from the **root** of a public repo, so this
folder can't be submitted from here:

1. Fork or create a public repo from heads-up-bundle.
2. Copy this folder's contents into its root (this note can stay or go).
3. Optionally change `PARTICIPANT` (currently `Datathon26`) to your team name.
4. Commit, push, and submit the full 40-character commit ID from
   `git rev-parse HEAD`.

Run it locally (needs `pip install -r requirements-dev.txt`):

```bash
python agent.py                              # this bot, port 5008
python baseline-bots/baseline_random.py      # opponent, port 5009
python judge_engine.py --deals 1000          # on Windows add --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5009
```
