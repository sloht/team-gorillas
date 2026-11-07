# What's in the logs

House Edge / Slot Machine Forecasting

Each row of `queries.csv` asks for the next spin on one machine. Use its history and the histories of related machines to predict five payout probabilities.

| File | Contents |
| --- | --- |
| queries.csv | query_id, episode_id, machine_id, machine_type, signal, cutoff |
| history.csv | episode_id, machine_id, tick, payout |
| labels.csv (train/validation) | Observed next-spin payout for each query |
| conditions.csv (train/validation) | Condition labels for comparing results |
| targets.csv (validation) | True probabilities and sampled outcome |
| reference.csv (validation) | Frozen pooled starter forecasts |
| metadata.json | Data version, round, season ID, points, and dataset sizes |

Test data contains queries and observed histories. Test outcomes, probabilities, conditions, and reference forecasts stay with organizers.

## Reading a machine's history

An episode groups related machines at a common prediction cutoff. A machine ID belongs to that episode. Types are `steady`, `balanced`, and `jackpot`; the observable `signal` is -1, 0, or 1.

Cutoffs range from 192 to 256 ticks. Long histories include every earlier tick. Short histories include 40 recent consecutive ticks and 24 sampled older ticks. A missing tick is unknown, not a loss.

The payout is the gross return on a one-chip wager: 0 loses the chip, 1 returns it, and 2, 5, or 20 returns that many chips.

## Patterns to look for

Past outcomes can interact at delays up to eight ticks. Two machines of the same type can react differently or oppositely. Useful patterns can repeat across machines, while their direction differs.

Both the chance of winning and the sizes of positive payouts can change. Some relationships gradually reverse; others change together across an episode. Earlier outcomes on neighboring machines can help, and hidden state leaves some uncertainty.

Ticks are generated synchronously from earlier outcomes. Another machine's result at the tick you're predicting is not an available input. When building training examples from history, a future observation can be a label for its earlier prediction, but never a feature for that prediction.

## The second round

Both rounds contain stable, short-history, drift, and shared-change episodes. Adaptation changes which temporal relationships recur and supplies fresh training and validation examples. Main and adaptation have separate training, validation, and test episodes with disjoint IDs.

The kit's data version is `experimentation-suite-v3`. Main lives in `season/main/public`; adaptation lives in `season/adaptation/public` and is released after main closes.
