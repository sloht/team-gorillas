# Adaptive lag forecast with episode shift and short-history shrink

This entry trains a pure-Python categorical forecaster from all observed training
history ticks, using outcomes at earlier ticks as labels and only earlier
observations as features.

Base model (`adaptive_lags`): it conditions on outcomes at lags 1, 2, 4, and 8,
blends that estimate with local payout frequencies at half-lives 8, 24, and
100, and adapts the short/long blend when their distributions disagree.

Two corrections are layered on top (`slot_forecasting/advanced2.py`):

1. **Episode shift.** For each earlier tick of the same machine in the same
   episode, the residual is the observed payout minus the base forecast for that
   tick. A recency-weighted mean residual (half-life 32 ticks, 128-tick window,
   shrunk with a pseudo-count of 80 and a gain of 0.7) is added to the forecast,
   clipped at zero, and renormalized.
2. **Short-history shrink.** For jackpot and balanced machines with fewer than
   128 observed ticks before the query cutoff, the forecast is blended 50/50
   with the pooled starter frequencies.

Leak rules: a query with cutoff C uses only ticks before C from its own
episode. A missing tick is unknown, never a zero payout. No condition labels are
used; the observed tick count and machine type are the only gating signals.

Settings were chosen on the tuning half of the validation episodes. On the
validation data, this model scored 31.346 (tuning half), 31.035 (audit half), and
31.203 (full) points, versus 31.245, 30.919, and 31.094 for the plain
`adaptive_lags` model. Hidden test performance is not guaranteed.

Run with Python 3.13 and the standard library:

```sh
python predict.py --data /path/to/public --out /path/to/submission.csv
```

The command reads `train/` and `test/` under the supplied public-data folder.
It does not use validation targets or test answers. The test forecast is
uploaded separately from this code archive.

Optional: `--split validation` predicts the validation split instead, which can
be compared with the scored candidate predictions.
