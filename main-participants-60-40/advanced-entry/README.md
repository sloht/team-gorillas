# Adaptive lag and multiscale forecast

This entry trains a pure-Python categorical forecaster from all observed training
history ticks, using outcomes at earlier ticks as labels and only earlier
observations as features. It conditions on outcomes at lags 1, 2, 4, and 8,
blends that estimate with local payout frequencies at half-lives 8, 24, and
100, and adapts the short/long blend when their distributions disagree.

Run with Python 3.13 and the standard library:

```sh
python predict.py --data /path/to/public --out /path/to/submission.csv
```

The command reads `train/` and `test/` under the supplied public-data folder.
It does not use validation targets or test answers. The test forecast is
uploaded separately from this code archive.
