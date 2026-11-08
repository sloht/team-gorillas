"""Round-two candidate forecasters. New file: advanced.py and the existing models are unchanged.

Leak rules for every feature in this file:
  * A query with cutoff C uses only ticks < C.
  * Only the same episode is used.
  * A missing tick is unknown (category -1), never a zero payout.

run_candidates.py picks up candidates() from this module automatically.
"""
from collections import Counter, defaultdict

from .advanced import AllTicksForecaster, _normalize
from .baseline import FrequencyForecaster
from .schema import PAYOUTS, ValidationError, index_history

COHORTS = ("self", "type", "type_signal")
N = len(PAYOUTS)


def _adaptive_lags():
    return AllTicksForecaster("lags", strength=24, local_weight=0.40,
                              half_lives=(8, 24, 100), adaptive=True)


class ShortHistoryShrink:
    """adaptive_lags, with extra weight on the pooled starter for machines with few observed ticks.

    The observed-tick count is a test-time signal; condition labels are not used.
    The threshold is an absolute tick count, so check it if a new round changes history lengths.
    """

    def __init__(self, threshold=128, weight=0.3):
        if threshold < 1 or not 0 <= weight <= 1:
            raise ValidationError("threshold must be at least 1 and weight in [0, 1].")
        self.threshold = int(threshold)
        self.weight = float(weight)
        self.model = _adaptive_lags()
        self.pooled = FrequencyForecaster("pooled")

    def fit(self, queries, history, labels=None):
        queries, history = tuple(queries), tuple(history)
        self.model.fit(queries, history, labels)
        self.pooled.fit(queries, history, labels)
        return self

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        main = self.model.predict(queries, history)
        pooled = self.pooled.predict(queries, history)
        counts = Counter((row.episode_id, row.machine_id) for row in history)
        result = {}
        for query in queries:
            first, second = main[query.query_id], pooled[query.query_id]
            if counts.get((query.episode_id, query.machine_id), 0) < self.threshold:
                weight = self.weight
                result[query.query_id] = tuple((1 - weight) * a + weight * b
                                               for a, b in zip(first, second))
            else:
                result[query.query_id] = tuple(first)
        return result


class GatedShrink:
    """Wrap any base forecaster and blend in the pooled starter for short-history machines.

    Differences from ShortHistoryShrink:
      * The base model is supplied by the caller (for example a shift forecaster).
      * machine_types (optional) restricts the blend to those machine types; None means all.
      * The observed-tick count only includes ticks < query.cutoff for the same episode and machine.

    No condition labels are used; the observed tick count and machine_type are query/history signals.
    """

    def __init__(self, base, threshold=128, weight=0.3, machine_types=None):
        if threshold < 1 or not 0 <= weight <= 1:
            raise ValidationError("threshold must be at least 1 and weight in [0, 1].")
        self.base = base
        self.threshold = int(threshold)
        self.weight = float(weight)
        self.machine_types = None if machine_types is None else frozenset(machine_types)
        self.pooled = FrequencyForecaster("pooled")

    def fit(self, queries, history, labels=None):
        queries, history = tuple(queries), tuple(history)
        fitted = self.base.fit(queries, history, labels)
        if fitted is not None:
            self.base = fitted
        self.pooled.fit(queries, history, labels)
        return self

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        main = self.base.predict(queries, history)
        pooled = self.pooled.predict(queries, history)
        ticks = defaultdict(list)
        for row in history:
            ticks[(row.episode_id, row.machine_id)].append(row.tick)
        result = {}
        for query in queries:
            first, second = main[query.query_id], pooled[query.query_id]
            gated = self.machine_types is None or query.machine_type in self.machine_types
            observed = sum(1 for tick in ticks.get((query.episode_id, query.machine_id), ())
                           if tick < query.cutoff)
            if gated and observed < self.threshold:
                weight = self.weight
                result[query.query_id] = tuple((1 - weight) * a + weight * b
                                               for a, b in zip(first, second))
            else:
                result[query.query_id] = tuple(first)
        return result


class EpisodeShiftForecaster(AllTicksForecaster):
    """adaptive_lags plus a shrunken, recency-weighted episode-level residual shift.

    For every earlier tick t < cutoff of each cohort machine in the same episode,
    residual = onehot(actual payout) - lag-conditional forecast for that tick.
    The weighted mean residual (half_life, shrunk by a pseudo-count) is added to the
    base forecast, which is then clipped at zero and renormalized.

    cohort: "self" (the machine itself), "type" (same machine_type), or
    "type_signal" (same machine_type and signal).
    """

    def __init__(self, cohort="type", half_life=32.0, shrink=40.0, gain=1.0, window=128,
                 strength=24.0, local_weight=0.40, half_lives=(8.0, 24.0, 100.0), adaptive=True):
        if cohort not in COHORTS:
            raise ValidationError(f"cohort must be one of {COHORTS}.")
        if half_life <= 0 or shrink <= 0 or gain < 0 or window < 1:
            raise ValidationError("half_life and shrink must be positive, gain nonnegative, window at least 1.")
        super().__init__("lags", strength=strength, local_weight=local_weight,
                         half_lives=half_lives, adaptive=adaptive)
        self.cohort = cohort
        self.shift_half_life = float(half_life)
        self.shrink = float(shrink)
        self.gain = float(gain)
        self.window = int(window)

    def _conditional(self, machine_key, feature_key):
        prior = self.priors.get(machine_key, self.base.global_prior)
        observed = self.counts.get((machine_key, feature_key))
        if observed is None:
            return prior
        return _normalize([count + self.strength * value for count, value in zip(observed, prior)])

    def _machine_residuals(self, rows, query):
        """(tick, residual vector) for every observed tick of one machine; uses only earlier ticks as context."""
        sequence = {row.tick: row.payout for row in rows}
        machine_key = self._machine_key(query)
        result = []
        for row in rows:
            feature_key = self._feature_key(sequence, query.machine_id, row.tick, {})
            conditional = self._conditional(machine_key, feature_key)
            result.append((row.tick, tuple((1.0 if row.payout == value else 0.0) - probability
                                           for value, probability in zip(PAYOUTS, conditional))))
        return result

    def _in_cohort(self, query, other):
        if self.cohort == "self":
            return other.machine_id == query.machine_id
        if other.machine_type != query.machine_type:
            return False
        return self.cohort == "type" or other.signal == query.signal

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        result = dict(super().predict(queries, history))
        indexed = index_history(queries, history)
        by_episode = defaultdict(list)
        for query in queries:
            by_episode[query.episode_id].append(query)
        cache = {}

        def residuals(other):
            key = (other.episode_id, other.machine_id)
            if key not in cache:
                cache[key] = self._machine_residuals(indexed.get(key, []), other)
            return cache[key]

        for query in queries:
            total = [0.0] * N
            weight_sum = 0.0
            low = query.cutoff - self.window
            for other in by_episode[query.episode_id]:
                if not self._in_cohort(query, other):
                    continue
                for tick, residual in residuals(other):
                    # Ticks at or after this query's cutoff are never used, even for other machines.
                    if tick >= query.cutoff or tick < low:
                        continue
                    weight = 2.0 ** (-(query.cutoff - 1 - tick) / self.shift_half_life)
                    weight_sum += weight
                    for index in range(N):
                        total[index] += weight * residual[index]
            if weight_sum <= 0:
                continue
            scale = self.gain / (weight_sum + self.shrink)
            raw = [max(0.0, probability + scale * shift)
                   for probability, shift in zip(result[query.query_id], total)]
            result[query.query_id] = _normalize(raw)
        return result


JACKPOT_BALANCED = ("jackpot", "balanced")


def _shift(cohort, gain, shrink):
    return EpisodeShiftForecaster(cohort, half_life=32.0, shrink=float(shrink), gain=float(gain))


def candidates():
    result = []

    # 1. Episode shift grid: cohort x gain x shrink, half_life fixed at 32.
    for cohort in ("self", "type"):
        for gain in (0.3, 0.5, 0.7):
            for shrink in (40, 80):
                name = f"shift_{cohort}_g{gain}_s{shrink}"
                result.append((name, lambda c=cohort, g=gain, s=shrink: _shift(c, g, s)))

    # 2. Short-history shrink toward the pooled starter, all machine types.
    for weight in (0.3, 0.5):
        result.append((f"shrink_short_{weight}",
                       lambda w=weight: ShortHistoryShrink(threshold=128, weight=w)))

    # 3. Same shrink, applied only to jackpot and balanced machines.
    for weight in (0.3, 0.5):
        result.append((f"shrink_jb_{weight}",
                       lambda w=weight: GatedShrink(_adaptive_lags(), threshold=128, weight=w,
                                                    machine_types=JACKPOT_BALANCED)))

    # 4a. Mid-grid combos from the candidates-03 run (kept so that run stays reproducible).
    result.append(("combo_shift_type_g0.5_s40_shrink_0.3",
                   lambda: GatedShrink(_shift("type", 0.5, 40), threshold=128, weight=0.3)))
    result.append(("combo_shift_type_g0.5_s40_shrink_jb_0.3",
                   lambda: GatedShrink(_shift("type", 0.5, 40), threshold=128, weight=0.3,
                                       machine_types=JACKPOT_BALANCED)))

    # 4b. Combo of the tuning-best shift (shift_self_g0.7_s80, mean ratio 0.9165) and the
    # tuning-best shrink (shrink_jb_0.5, mean ratio 0.9171) from candidates-03.
    # Shift forecast first, then the jackpot/balanced pooled blend on top.
    result.append(("combo_shift_self_g0.7_s80_shrink_jb_0.5",
                   lambda: GatedShrink(_shift("self", 0.7, 80), threshold=128, weight=0.5,
                                       machine_types=JACKPOT_BALANCED)))
    return result
