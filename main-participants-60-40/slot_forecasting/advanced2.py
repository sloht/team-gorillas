"""Round-two candidate forecasters. New file: advanced.py and the existing models are unchanged.

Leak rules for every feature in this file:
  * A query with cutoff C uses only ticks < C.
  * Only the same episode is used.
  * A missing tick is unknown (category -1), never a zero payout.

run_candidates.py picks up candidates() from this module automatically.
"""
from collections import Counter, defaultdict
import math
import sys

from .advanced import AllTicksForecaster, _category, _normalize
from .baseline import FrequencyForecaster
from .schema import PAYOUTS, ValidationError, index_history

COHORTS = ("self", "type", "type_signal")
N = len(PAYOUTS)
LAG_CANDIDATES = tuple(range(1, 9))


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


def _new_lag_bucket():
    """Accumulator for lag scoring: marginal payout counts and, per lag, payout counts per lag category."""
    return {"marginal": [0.0] * N,
            "lags": {lag: defaultdict(lambda: [0.0] * N) for lag in LAG_CANDIDATES}}


class LagSelectedForecaster(AllTicksForecaster):
    """AllTicksForecaster whose temporal lags are chosen per machine_type in fit().

    For each machine_type (and one all-types bucket used as a fallback for unseen types),
    every lag in 1..8 is scored on the training history. Each observed tick t contributes
    the next payout (one-hot) and the payout category at tick t - lag, where a missing tick
    is the unknown category -1, never a zero payout. The score is the drop in squared error
    when the payout distribution is predicted from that category instead of the type's
    marginal distribution: sum over categories of n_c * ||p_c - p||^2, divided by n.
    The top_k lags (ties go to the smaller lag) become that type's lag set.

    Only the supplied history is used, which index_history() already restricts to ticks
    before each machine's cutoff, within one episode. No condition labels are used.
    Selected lags and gains are printed to stderr during fit() and kept on
    selected_lags / lag_gains.
    """

    def __init__(self, top_k=4, strength=24.0, local_weight=0.40,
                 half_lives=(8.0, 24.0, 100.0), adaptive=True, label="lag_selected", verbose=True):
        if not 1 <= top_k <= len(LAG_CANDIDATES):
            raise ValidationError(f"top_k must be between 1 and {len(LAG_CANDIDATES)}.")
        super().__init__("lags", strength=strength, local_weight=local_weight,
                         half_lives=half_lives, adaptive=adaptive)
        self.top_k = int(top_k)
        self.label = str(label)
        self.verbose = bool(verbose)
        self.selected_lags = {}
        self.lag_gains = {}
        self.default_lags = (1, 2, 4, 8)[:self.top_k]
        self.default_gains = {}

    @staticmethod
    def _gains(bucket):
        total = math.fsum(bucket["marginal"])
        if total <= 0:
            return {lag: 0.0 for lag in LAG_CANDIDATES}
        overall = [value / total for value in bucket["marginal"]]
        gains = {}
        for lag in LAG_CANDIDATES:
            explained = 0.0
            for counts in bucket["lags"][lag].values():
                n = math.fsum(counts)
                if n <= 0:
                    continue
                explained += n * math.fsum((count / n - p) ** 2 for count, p in zip(counts, overall))
            gains[lag] = explained / total
        return gains

    def _select_lags(self, indexed, query_by_machine):
        buckets = defaultdict(_new_lag_bucket)
        everything = _new_lag_bucket()
        for key, rows in indexed.items():
            query = query_by_machine[key]
            sequence = {row.tick: row.payout for row in rows}
            bucket = buckets[query.machine_type]
            for row in rows:
                outcome = PAYOUTS.index(row.payout)
                for target in (bucket, everything):
                    target["marginal"][outcome] += 1.0
                for lag in LAG_CANDIDATES:
                    category = _category(sequence.get(row.tick - lag))
                    bucket["lags"][lag][category][outcome] += 1.0
                    everything["lags"][lag][category][outcome] += 1.0

        def top(gains):
            ranked = sorted(LAG_CANDIDATES, key=lambda lag: (-gains[lag], lag))
            return tuple(sorted(ranked[:self.top_k]))

        self.selected_lags, self.lag_gains = {}, {}
        for machine_type, bucket in buckets.items():
            gains = self._gains(bucket)
            self.lag_gains[machine_type] = gains
            self.selected_lags[machine_type] = top(gains)
        if math.fsum(everything["marginal"]) > 0:
            self.default_gains = self._gains(everything)
            self.default_lags = top(self.default_gains)

    def _report(self):
        def line(name, lags, gains):
            detail = " ".join(f"{lag}={gains[lag]:.5f}" for lag in LAG_CANDIDATES)
            print(f"[{self.label}] machine_type={name} lags={lags} gains: {detail}", file=sys.stderr)

        for machine_type in sorted(self.selected_lags):
            line(machine_type, self.selected_lags[machine_type], self.lag_gains[machine_type])
        if self.default_gains:
            line("(all types)", self.default_lags, self.default_gains)

    def _lags_for(self, machine_type):
        return self.selected_lags.get(machine_type, self.default_lags)

    @staticmethod
    def _lag_key(sequence, tick, lags):
        # A missing tick maps to category -1 (unknown), never to a zero payout.
        return tuple(_category(sequence.get(tick - lag)) for lag in lags), None

    def fit(self, queries, history, labels=None):
        queries, history = tuple(queries), tuple(history)
        indexed = index_history(queries, history)
        self.base.fit(queries, history, labels)
        query_by_machine = {(q.episode_id, q.machine_id): q for q in queries}
        self._select_lags(indexed, query_by_machine)
        if self.verbose:
            self._report()

        self.counts = defaultdict(lambda: [0.0] * N)
        global_counts = [0.5] * N
        context_counts = defaultdict(lambda: [0.5] * N)
        for (episode, machine_id), rows in indexed.items():
            query = query_by_machine[(episode, machine_id)]
            sequence = {row.tick: row.payout for row in rows}
            machine_key = self._machine_key(query)
            lags = self._lags_for(query.machine_type)
            for row in rows:
                key = self._lag_key(sequence, row.tick, lags)
                index = PAYOUTS.index(row.payout)
                self.counts[(machine_key, key)][index] += 1.0
                context_counts[machine_key][index] += 1.0
                global_counts[index] += 1.0
        global_prior = _normalize(global_counts)
        self.priors = {
            key: _normalize([count + 24.0 * prior for count, prior in zip(values, global_prior)])
            for key, values in context_counts.items()
        }
        return self

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        indexed = index_history(queries, history)
        result = {}
        for query in queries:
            rows = indexed.get((query.episode_id, query.machine_id), [])
            sequence = {row.tick: row.payout for row in rows}
            feature_key = self._lag_key(sequence, query.cutoff, self._lags_for(query.machine_type))
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


class BlendForecaster:
    """Convex blend (1 - weight) * first + weight * second of any two forecasters."""

    def __init__(self, first, second, weight=0.3):
        if not 0 <= weight <= 1:
            raise ValidationError("weight must be in [0, 1].")
        self.first = first
        self.second = second
        self.weight = float(weight)

    def fit(self, queries, history, labels=None):
        queries, history = tuple(queries), tuple(history)
        fitted = self.first.fit(queries, history, labels)
        if fitted is not None:
            self.first = fitted
        fitted = self.second.fit(queries, history, labels)
        if fitted is not None:
            self.second = fitted
        return self

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        first = self.first.predict(queries, history)
        second = self.second.predict(queries, history)
        weight = self.weight
        return {query_id: tuple((1 - weight) * a + weight * b
                                for a, b in zip(values, second[query_id]))
                for query_id, values in first.items()}


MIN_LIFT_RELIABILITY = 0.2
LIFT_LAGS = LAG_CANDIDATES
PAYOUT_ZERO_INDEX = PAYOUTS.index(0)


def _paid(payout):
    return 1.0 if payout > 0 else 0.0


def _lag_stats(sequence, targets, lag, cutoff=None, half_life=None):
    """Paid lift of one machine at one lag, using only pairs (t, t - lag) where both ticks are observed.

    sequence maps observed tick -> payout; targets are the ticks t used as outcomes.
    lift = P(paid at t | paid at t - lag) - P(paid at t | not paid at t - lag).
    q is the paid fraction of the lagged tick over the same pairs.
    With half_life (and cutoff), each pair is weighted 2 ** (-(cutoff - 1 - t) / half_life).
    Returns (lift, q, n_eff, n_pairs), or None when either lagged group is empty.
    """
    weight_total = weight_squares = 0.0
    weight_x1 = weight_x0 = paid_x1 = paid_x0 = 0.0
    pairs = 0
    for tick in targets:
        lagged = sequence.get(tick - lag)
        if lagged is None:
            continue
        if half_life is None or cutoff is None:
            weight = 1.0
        else:
            weight = 2.0 ** (-(cutoff - 1 - tick) / half_life)
        outcome = _paid(sequence[tick])
        pairs += 1
        weight_total += weight
        weight_squares += weight * weight
        if _paid(lagged):
            weight_x1 += weight
            paid_x1 += weight * outcome
        else:
            weight_x0 += weight
            paid_x0 += weight * outcome
    if pairs == 0 or weight_x1 <= 0 or weight_x0 <= 0:
        return None
    lift = paid_x1 / weight_x1 - paid_x0 / weight_x0
    q = weight_x1 / weight_total
    n_eff = weight_total * weight_total / weight_squares
    return lift, q, n_eff, pairs


def _pearson(xs, ys):
    if len(xs) < 3:
        return 0.0
    mean_x, mean_y = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    cov = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = math.fsum((x - mean_x) ** 2 for x in xs)
    var_y = math.fsum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


class LagLiftForecaster:
    """Base forecast plus a shrunken, machine-specific paid-lift correction at reliable lags.

    fit(): fits the base, then for each machine_type scores every lag in 1..8 by split-half
    reliability of the per-machine paid lift on the training histories:
      * odd/even: lift from outcome ticks t with odd tick vs even tick;
      * first/second: lift from the first half vs the second half of the machine's observed ticks.
    Reliability is the Pearson correlation across machines of the same type between the two
    halves (a machine counts only if both halves have at least min_obs pairs and a defined lift).
    Score = min(odd/even, first/second). The top_k lags with score >= 0.2 are kept per type.

    predict(): for each selected lag L of the query's type, the machine's own lift (optionally
    recency-weighted by half_life) and paid fraction q use only pairs where both ticks are
    observed. lift is shrunk by n_eff / (n_eff + k). If tick cutoff - L is observed,
    delta += lift * (paid(cutoff - L) - q); otherwise that lag is skipped.
    new_paid = clip(base_paid + delta, 0.02, 0.98); probabilities of payouts > 0 are scaled by
    new_paid / base_paid, p0 = 1 - new_paid, then renormalized.

    Leak rules: only ticks < cutoff (index_history enforces this), same episode, missing ticks
    are unknown rather than zero, and no condition labels are used.
    Selected lags are printed to stderr during fit() and kept on selected_lags / reliability.
    """

    def __init__(self, base=_adaptive_lags, top_k=2, k=20.0, half_life=None, min_obs=20,
                 label="lag_lift", verbose=True):
        if not 1 <= top_k <= 3:
            raise ValidationError("top_k must be between 1 and 3.")
        if k < 0:
            raise ValidationError("k must be nonnegative.")
        if half_life is not None and half_life <= 0:
            raise ValidationError("half_life must be None or positive.")
        if min_obs < 1:
            raise ValidationError("min_obs must be at least 1.")
        self.base_factory = base
        self.base = None
        self.top_k = int(top_k)
        self.k = float(k)
        self.half_life = None if half_life is None else float(half_life)
        self.min_obs = int(min_obs)
        self.label = str(label)
        self.verbose = bool(verbose)
        self.selected_lags = {}
        self.reliability = {}

    def _split_lifts(self, sequence, lag):
        """Lifts (or None) for the odd, even, first-half and second-half outcome ticks of one machine."""
        ticks = sorted(sequence)
        middle = len(ticks) // 2
        splits = {
            "odd": [t for t in ticks if t % 2 == 1],
            "even": [t for t in ticks if t % 2 == 0],
            "first": ticks[:middle],
            "second": ticks[middle:],
        }
        lifts = {}
        for name, targets in splits.items():
            stats = _lag_stats(sequence, targets, lag)
            lifts[name] = stats[0] if stats is not None and stats[3] >= self.min_obs else None
        return lifts

    def _select_lags(self, indexed, query_by_machine):
        pairs = defaultdict(lambda: {lag: {"oe": ([], []), "fs": ([], [])} for lag in LIFT_LAGS})
        for key, rows in indexed.items():
            machine_type = query_by_machine[key].machine_type
            sequence = {row.tick: row.payout for row in rows}
            for lag in LIFT_LAGS:
                lifts = self._split_lifts(sequence, lag)
                bucket = pairs[machine_type][lag]
                if lifts["odd"] is not None and lifts["even"] is not None:
                    bucket["oe"][0].append(lifts["odd"])
                    bucket["oe"][1].append(lifts["even"])
                if lifts["first"] is not None and lifts["second"] is not None:
                    bucket["fs"][0].append(lifts["first"])
                    bucket["fs"][1].append(lifts["second"])

        self.selected_lags, self.reliability = {}, {}
        for machine_type, per_lag in pairs.items():
            scores = {}
            for lag in LIFT_LAGS:
                odd_even = _pearson(*per_lag[lag]["oe"])
                first_second = _pearson(*per_lag[lag]["fs"])
                scores[lag] = (odd_even, first_second, min(odd_even, first_second))
            self.reliability[machine_type] = scores
            ranked = sorted(LIFT_LAGS, key=lambda lag: (-scores[lag][2], lag))
            chosen = [lag for lag in ranked[:self.top_k] if scores[lag][2] >= MIN_LIFT_RELIABILITY]
            self.selected_lags[machine_type] = tuple(sorted(chosen))

    def _report(self):
        for machine_type in sorted(self.selected_lags):
            scores = self.reliability[machine_type]
            detail = " ".join(f"{lag}={scores[lag][2]:.3f}" for lag in LIFT_LAGS)
            print(f"[{self.label}] machine_type={machine_type} lags={self.selected_lags[machine_type]} "
                  f"score(min of odd/even, first/second): {detail}", file=sys.stderr)

    def fit(self, queries, history, labels=None):
        queries, history = tuple(queries), tuple(history)
        indexed = index_history(queries, history)
        self.base = self.base_factory()
        fitted = self.base.fit(queries, history, labels)
        if fitted is not None:
            self.base = fitted
        query_by_machine = {(q.episode_id, q.machine_id): q for q in queries}
        self._select_lags(indexed, query_by_machine)
        if self.verbose:
            self._report()
        return self

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        result = dict(self.base.predict(queries, history))
        indexed = index_history(queries, history)
        for query in queries:
            lags = self.selected_lags.get(query.machine_type, ())
            if not lags:
                continue
            rows = indexed.get((query.episode_id, query.machine_id), [])
            sequence = {row.tick: row.payout for row in rows}
            targets = sorted(sequence)
            base = result[query.query_id]
            base_paid = 1.0 - base[PAYOUT_ZERO_INDEX]
            if base_paid <= 0:
                continue
            delta = 0.0
            used = False
            for lag in lags:
                lagged = sequence.get(query.cutoff - lag)
                if lagged is None:
                    continue
                stats = _lag_stats(sequence, targets, lag, query.cutoff, self.half_life)
                if stats is None or stats[3] < self.min_obs:
                    continue
                lift, q, n_eff, _ = stats
                lift *= n_eff / (n_eff + self.k)
                delta += lift * (_paid(lagged) - q)
                used = True
            if not used:
                continue
            new_paid = min(0.98, max(0.02, base_paid + delta))
            scale = new_paid / base_paid
            adjusted = [(1.0 - new_paid) if index == PAYOUT_ZERO_INDEX else value * scale
                        for index, value in enumerate(base)]
            result[query.query_id] = _normalize(adjusted)
        return result


JACKPOT_BALANCED = ("jackpot", "balanced")


def _shift(cohort, gain, shrink):
    return EpisodeShiftForecaster(cohort, half_life=32.0, shrink=float(shrink), gain=float(gain))


def _grid_model(strength, local_weight):
    return AllTicksForecaster("lags", strength=strength, local_weight=local_weight,
                              half_lives=(8, 24, 100), adaptive=True)


def _blend_pooled_recent(weight):
    return BlendForecaster(_adaptive_lags(), FrequencyForecaster("pooled_recent"), weight)


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

    # ---- candidates-04 ----

    # 5. Lags chosen per machine_type from 1..8 in fit(); prints the selection to stderr.
    for top_k in (3, 4):
        result.append((f"lagsel_top{top_k}",
                       lambda k=top_k: LagSelectedForecaster(top_k=k, label=f"lagsel_top{k}")))

    # 6. Strength x local_weight grid on AllTicksForecaster (lags, adaptive, half_lives 8/24/100).
    # The (24, 0.4) cell is skipped: it is identical to the baseline adaptive_lags.
    for strength in (12, 24, 48):
        for local_weight in (0.3, 0.4, 0.5):
            if strength == 24 and local_weight == 0.4:
                continue
            result.append((f"grid_s{strength}_w{local_weight}",
                           lambda s=strength, w=local_weight: _grid_model(s, w)))

    # 7. Blend of adaptive_lags with the pooled_recent starter.
    for weight in (0.2, 0.4):
        result.append((f"blend_pooled_recent_{weight}",
                       lambda w=weight: _blend_pooled_recent(w)))

    # 8. Short-history shrink (threshold 128, weight 0.3, all machine types) on top of the new
    # structural candidates. GatedShrink with machine_types=None behaves like ShortHistoryShrink
    # for any base model. shrink_short_0.3 above already covers the plain adaptive_lags base.
    for top_k in (3, 4):
        result.append((f"shrink128_0.3_on_lagsel_top{top_k}",
                       lambda k=top_k: GatedShrink(
                           LagSelectedForecaster(top_k=k, label=f"shrink128_lagsel_top{k}", verbose=False),
                           threshold=128, weight=0.3)))
    for weight in (0.2, 0.4):
        result.append((f"shrink128_0.3_on_blend_pooled_recent_{weight}",
                       lambda w=weight: GatedShrink(_blend_pooled_recent(w), threshold=128, weight=0.3)))

    # 9. Shrink on top of the tuning-best grid cell from candidates-04 (grid_s12_w0.5,
    # tuning mean ratio 0.9152). Chosen by tuning only.
    result.append(("shrink128_0.3_on_grid_s12_w0.5",
                   lambda: GatedShrink(_grid_model(12, 0.5), threshold=128, weight=0.3)))

    # ---- candidates-05 ----

    # 10. Machine-specific paid-lift correction at reliable lags (chosen per machine_type by
    # split-half reliability on the training histories): top_k x k x half_life grid.
    for top_k in (1, 2, 3):
        for shrink_k in (10, 20, 40):
            for half_life in (None, 100):
                suffix = "none" if half_life is None else str(half_life)
                name = f"liftlag_top{top_k}_k{shrink_k}_hl{suffix}"
                result.append((name, lambda t=top_k, s=shrink_k, h=half_life, n=name: LagLiftForecaster(
                    top_k=t, k=s, half_life=h, label=n)))
    return result
