# House rules

House Edge / Slot Machine Forecasting / TAMU Datathon 2026

Predict the next spin's probabilities of returning **0×, 1×, 2×, 5×, or 20×** a one-chip wager. The machines and gameplay logs are simulated for this challenge.

## Two rounds

**Main round — 60 points.** Learn from the main training and validation data, then submit forecasts for its test queries.

**Round two (adaptation) — 40 points.** Main submissions close and the second kit opens **Saturday, November 7, 2026 at 9 PM Central (America/Chicago)**. The new kit contains training data, validation answers, and test histories with changed patterns. Round two closes **Sunday, November 8 at 11 AM Central**, giving everyone **14 hours**. You may reuse your code and main-round data, refit the same model, or try another approach.

Both rounds score your forecasts directly. Round two doesn't give a bonus for improving on your main score. Organizers will share a Google Form for each round and explain how upload problems will be handled. Attach both files and click Submit before the deadline; the Form timestamp determines whether a response arrived in time. Validation happens afterward.

## How points work

For each of the four conditions, average squared probability error within each episode, then average the episodes. Call this error `E[c]`. Calculate `B[c]` the same way for the frozen pooled starter fitted on that round's training data.

```
R = mean over four conditions of E[c] / max(B[c], 1e-9)
main points = 60 / (1 + R_main)
adaptation points = 40 / (1 + R_adaptation)
final points = main points + adaptation points
```

The conditions are stable, short history, drift, and shared change. Each counts equally within its round. Main and adaptation have separate frozen references. The pooled starter normally scores 30 + 20 = 50; exact knowledge of all probabilities scores 100. These are competition points, not accuracy percentages.

Forecasts are compared with the simulator's true conditional probabilities, so sampled jackpots do not decide the ranking. Some machine state is hidden, so perfect knowledge is generally unavailable. Your probabilities should reflect what the observed history can tell you.

All points come from forecasts. To qualify for awards, your code must recreate your predictions and follow the data rules. Equal totals share a rank; organizers will announce how prize ties are handled before kickoff.

## At the table

- Use organizer-released data. Main data may be reused during adaptation. Test answers and private organizer files are off limits.
- Use observations strictly before the prediction tick, including when making training examples from earlier history. Missing ticks are unobserved, not zero payouts.
- Earlier outcomes can interact at lags up to eight ticks. Machines of the same type can respond differently or oppositely; both win chances and winning amounts can change. Other machines' earlier outcomes may help. Condition labels are supplied for training/validation only.
- Use any model family, model-selection tools, or coding assistants. You are responsible for a working, reproducible submission.
- Your submitted code must make its predictions without internet access or external APIs. Use the runtime and dependency instructions in the participant guide.
- All teams receive the same files and deadlines. Final-test feedback is withheld while submissions are open. Public validation is available for model selection.

## Submit your forecast

For each round, submit one CSV with exactly these columns:

```csv
query_id,p_0,p_1,p_2,p_5,p_20
```

Include every test query once. Probabilities must be finite, between zero and one, and sum to one within `1e-6`. Include your code/model, dependency versions, seeds/settings, and one command that recreates the CSV. A short setup README is enough.

The kit includes `check_submission.py`, `build_entry.py`, and `SUBMISSION_README.md`. Submit the CSV (up to 10 MB) and code ZIP (up to 100 MB) through the round's Google Form, choosing your team from its list. The ZIP must open directly to `README.md` and `reproduce.json`; the participant guide explains the setup and limits. Running the local checker doesn't submit anything. Wait for the Form confirmation and keep the exact files. Form links will be announced. MB means 1,000,000 bytes.

Your team's latest Form response received before each deadline counts, even if its files are invalid. Main entries can't be replaced after round two opens. A missing or invalid round-two entry earns zero for that round. You need a valid main entry for an overall ranking.

A replacement still counts if validation or its code check later fails; we don't automatically switch back to an older entry. Organizers validate submissions after cutoff and rerun prize contenders' code before awards. Keep `"image": "python"` and pin extra packages in a root-level `requirements.txt`; organizers prepare those dependencies before an offline prediction run. Runtime limits are 30 minutes, two CPU cores, and 16 GiB RAM, with a maximum probability difference of `1e-8` after normalization. Resolve organizer-side failures before final rankings. Any changes must be announced consistently to all teams.
