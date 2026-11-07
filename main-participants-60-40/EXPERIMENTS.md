# Make your forecasts better

Get the starter working first. Then change one thing and compare the score on data you didn't train on. A more complicated model is useful only if it makes better predictions.

## Start with small changes

| Try | What to do | What to check |
| --- | --- | --- |
| Recent history | Compare the last 20, 50, or 100 observed plays, or give recent plays more weight. | Does it handle changes better without losing too much on steady machines? |
| Each machine's habits | Make forecasts per machine; mix in a broader average when history is short. | Does this beat treating every machine alike? |
| Sequences | Use pairs of earlier payouts as inputs, rather than only counting payouts. | Does the pair help on fresh examples, beyond either payout alone? |
| Different memory lengths | Try a short window and a long window, or blend their forecasts. | Does the blend improve across more than one kind of machine? |

Keep predicting all five payout chances. Predicting only win/loss leaves out the size of the payout. Missing ticks stay unknown, and every input must come from before the spin being predicted.

## Go further if the basics work

- **Learn patterns per machine.** A useful pair of earlier spins may differ between machines. Try choosing or blending patterns separately.
- **Try a small sequence model.** A neural network could learn combinations of earlier payouts. Compare it with your simpler model using the same data split and training budget.
- **Combine models.** Average probability forecasts from approaches that make different mistakes. Choose the blend on validation data.
- **Prepare for the second round.** Keep an update/refit command and a simple fallback. The new data releases Saturday at 9 PM Central, with submissions due Sunday at 11 AM Central. Use the 14-hour window to compare changes and leave time for packaging and upload.

These are options, not requirements or guaranteed improvements. Start with a small experiment before committing the weekend to a bigger model. Model size and architecture don't earn points by themselves.

## Compare without fooling yourself

Use training data to fit the model, validation data to choose settings, and a separate check to confirm the improvement. If you keep tuning against that check, it stops being a fresh check. Final test answers stay hidden.

Compare round points and the scores for each machine condition. Keep a change that holds up across fresh examples, rather than one that only improves a few lucky cases. Save the code, settings, and command for your best run.

## Optional: use the experiment runner

```sh
python run_experiments.py --out experiments-01
```

It tries 12 small model settings. Half the validation episodes in each condition choose a model; the other half checks it afterward. It saves `submission.csv`, validation predictions, and `experiments.json` with the results.

Edit `starter_candidates()` in `slot_forecasting/experiments.py` to change the list. Or pass `(name, factory)` pairs to `run_experiments`. Each model needs `fit(queries, history, labels)` and `predict(queries, history)`. `TransitionForecaster` is a starting point for changing history lags, memory, smoothing, and features.

To check and score your own validation file from the released kit:

```sh
python check_submission.py my_validation.csv --split validation
```

This reports round points and condition loss ratios. For your test CSV, omit `--split validation`; test answers are private, so that checks format only. Neither command uploads your file.

A reasonable weekend plan: get one valid submission early, spend most of your time testing a few ideas, and leave time to reproduce your best result.
