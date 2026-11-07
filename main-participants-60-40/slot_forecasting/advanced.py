"""Pure-Python experimental forecasters using historical ticks as labels."""

from collections import defaultdict
import math

from .baseline import FrequencyForecaster
from .schema import PAYOUTS, ValidationError, index_history


def _category(value):
    if value is None:
        return -1
    return PAYOUTS.index(value)


def _normalize(values):
    total = math.fsum(values)
    if total <= 0:
        raise ValidationError("Cannot normalize an empty payout distribution.")
    return tuple(value / total for value in values)


class AllTicksForecaster:
    """Backoff conditional counts trained from each observed historical tick."""

    def __init__(self, feature="lag1", strength=24.0, local_weight=0.0,
                 half_lives=(8.0, 24.0, 100.0), adaptive=False):
        if feature not in ("lag1", "lags", "episode", "lags_episode"):
            raise ValidationError("Unknown all-ticks feature set.")
        if strength <= 0 or local_weight < 0:
            raise ValidationError("strength must be positive and local_weight nonnegative.")
        if not half_lives or any(value <= 0 for value in half_lives):
            raise ValidationError("half_lives must contain positive values.")
        self.feature = feature
        self.strength = float(strength)
        self.local_weight = float(local_weight)
        self.half_lives = tuple(float(value) for value in half_lives)
        self.adaptive = bool(adaptive)
        self.base = FrequencyForecaster("pooled_recent")
        self.counts = defaultdict(lambda: [0.0] * len(PAYOUTS))
        self.priors = {}

    @staticmethod
    def _machine_key(query):
        return query.machine_type, query.signal

    def _episode_feature(self, machine_id, tick, episode_timeline):
        values = [value for prior_tick in range(tick - 8, tick)
                  for other_id, value in episode_timeline.get(prior_tick, ())
                  if other_id != machine_id]
        if not values:
            return -1
        # Coarse bins keep the cross-machine context estimable from each episode.
        return min(3, int(math.fsum(values) / len(values) + 0.5))

    def _feature_key(self, sequence, machine_id, tick, episode_timeline):
        if self.feature in ("lag1", "lags"):
            lags = (1,) if self.feature == "lag1" else (1, 2, 4, 8)
            temporal = tuple(_category(sequence.get(tick - lag)) for lag in lags)
        else:
            temporal = ()
        if self.feature in ("episode", "lags_episode"):
            episode_context = self._episode_feature(machine_id, tick, episode_timeline)
        else:
            episode_context = None
        return temporal, episode_context

    def fit(self, queries, history, labels=None):
        queries, history = tuple(queries), tuple(history)
        indexed = index_history(queries, history)
        self.base.fit(queries, history, labels)
        global_counts = [0.5] * len(PAYOUTS)
        context_counts = defaultdict(lambda: [0.5] * len(PAYOUTS))
        query_by_machine = {(q.episode_id, q.machine_id): q for q in queries}
        episode_timeline = defaultdict(lambda: defaultdict(list))
        for row in history:
            episode_timeline[row.episode_id][row.tick].append((row.machine_id, _category(row.payout)))

        for (episode, machine_id), rows in indexed.items():
            query = query_by_machine[(episode, machine_id)]
            sequence = {row.tick: row.payout for row in rows}
            machine_key = self._machine_key(query)
            for row in rows:
                key = self._feature_key(sequence, machine_id, row.tick, episode_timeline[episode])
                self.counts[(machine_key, key)][PAYOUTS.index(row.payout)] += 1.0
                context_counts[machine_key][PAYOUTS.index(row.payout)] += 1.0
                global_counts[PAYOUTS.index(row.payout)] += 1.0
        global_prior = _normalize(global_counts)
        self.priors = {
            key: _normalize([count + 24.0 * prior for count, prior in zip(values, global_prior)])
            for key, values in context_counts.items()
        }
        return self

    def _local_frequency(self, query, rows):
        frequencies = []
        for half_life in self.half_lives:
            counts = [0.0] * len(PAYOUTS)
            for row in rows:
                age = query.cutoff - 1 - row.tick
                counts[PAYOUTS.index(row.payout)] += 2.0 ** (-age / half_life)
            frequencies.append(_normalize(counts) if math.fsum(counts) else self.base.predict((query,), rows)[query.query_id])
        if self.adaptive and len(frequencies) >= 2:
            short, long = frequencies[0], frequencies[-1]
            distance = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(short, long)))
            short_weight = min(0.8, max(0.25, 0.25 + 2.0 * distance))
            mixed = tuple(short_weight * a + (1 - short_weight) * b for a, b in zip(short, long))
        else:
            weights = tuple(1.0 / len(frequencies) for _ in frequencies)
            mixed = tuple(math.fsum(weight * values[i] for weight, values in zip(weights, frequencies))
                          for i in range(len(PAYOUTS)))
        return mixed

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        indexed = index_history(queries, history)
        episode_timeline = defaultdict(lambda: defaultdict(list))
        for row in history:
            episode_timeline[row.episode_id][row.tick].append((row.machine_id, _category(row.payout)))
        result = {}
        for query in queries:
            rows = indexed.get((query.episode_id, query.machine_id), [])
            sequence = {row.tick: row.payout for row in rows}
            feature_key = self._feature_key(sequence, query.machine_id, query.cutoff,
                                            episode_timeline[query.episode_id])
            machine_key = self._machine_key(query)
            prior = self.priors.get(machine_key, self.base.global_prior)
            observed = self.counts.get((machine_key, feature_key))
            if observed is None:
                conditional = prior
            else:
                conditional = _normalize([count + self.strength * value
                                          for count, value in zip(observed, prior)])
            if self.local_weight:
                local = self._local_frequency(query, rows)
                weight = self.local_weight
                conditional = tuple((1 - weight) * value + weight * local_value
                                    for value, local_value in zip(conditional, local))
            result[query.query_id] = conditional
        return result
