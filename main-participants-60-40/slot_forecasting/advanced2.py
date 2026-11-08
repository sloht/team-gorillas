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


def candidates():
    return [
        ("shrink_short_0.3", lambda: ShortHistoryShrink(threshold=128, weight=0.3)),
        ("shift_self_hl32", lambda: EpisodeShiftForecaster("self", half_life=32.0, shrink=40.0)),
        ("shift_type_hl32", lambda: EpisodeShiftForecaster("type", half_life=32.0, shrink=40.0)),
        ("shift_type_signal_hl32", lambda: EpisodeShiftForecaster("type_signal", half_life=32.0, shrink=40.0)),
    ]
