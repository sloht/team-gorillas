# Official tournament rules (do not edit)

`game.py`, `evaluator.py` and `judge_engine.py` are byte-for-byte copies from
https://github.com/tamu-datathon-org/heads-up-bundle at commit
`c83ab6eeb7660a2fd1fae7df224315d41fe985a9`.

| File | SHA-256 |
|---|---|
| `game.py` | `bc00e3c2d1bf64cbd36bf5553dae218767f11b20f388e5c417d989919877f2db` |
| `evaluator.py` | `e2a151a4c41828a035030f490c91b81015d04bc209abfb860ca3d0dce649e0b4` |
| `judge_engine.py` | `fa7a7e3fb49c09588cfcf0930a8e3f6c189e419dbc54263ae7d53b953223d422` |

Every evaluation in this repo (qualifying, the per-hand search, tournaments,
the evolution experiment and the play UI) referees hands with `game.Game`,
scores showdowns with `evaluator.omaha_score`, and scores matches with
`judge_engine.combine_halves`. Load them through `official_rules.py`, never by
copying their logic.

If the organisers update the bundle, replace all three files together, update
the commit and hashes above, and rerun `python3 -m unittest discover -s tests`.
`tests/test_official_rules.py` fails if these files differ from the hashes
listed here, so an accidental edit is caught.
