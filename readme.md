# Short Deck 5-Card Pot-Limit Omaha Bots

This project provides a small Python interface for poker bots and a heads-up
round-robin harness. No third-party packages are required.

## Rules used by the simulator

- Short Deck uses the 36 cards ranked 6 through Ace.
- Each player receives five private cards and must use exactly two of them with
  exactly three of the five community cards.
- Betting is pot-limit. A bet or raise action specifies the player's target
  contribution on the current street.
- A flush ranks above a full house. The Short Deck wheel is A-6-7-8-9.
- Heads-up matches play a fixed number of hands with equal starting stacks.
  Stacks reset after each hand, blinds stay fixed, and the button alternates.
  This avoids a player being eliminated before the requested hand count.
- Each hand number uses the same shuffled deck in every pairing, with button
  positions alternating, so matches compare strategies on paired deals.

## Run the built-in bot round robin

```bash
python3 run_tournament.py
```

By default, the equity-driven `ShortDeckPLOBot` competes with eight differently
styled reference strategies for 2,000 hands per matchup: random, calling
station, tight-aggressive, loose-aggressive, rock, maniac, pot-control, and
selective bluffing. Adjust the hand count, stack, blinds, and seed with
`--hands`, `--stack`, `--small-blind`, `--big-blind`, and `--seed`.

The standings rank by the number of distinct opponents beaten first, then
aggregate net chips. This prioritizes strategies that work across the whole
field over a large win against one narrow opponent. Each pairing uses paired
deals and alternating button positions to make comparisons fairer. For more
reliable comparisons, increase `--hands` and repeat with several `--seed`
values; a single run is not proof of a generally stronger strategy.

## Four-generation strategy experiment

```bash
python3 run_evolution.py --hands 100 --seed 42
```

The experiment runs four rounds of round robin. After each round, it keeps the
six strategies with the most opponents beaten (net chips break ties), retains
all eliminated bot code, and adds three new strategy classes. It then runs a
final round robin with all 21 strategies, including eliminated bots. Increase
`--hands` for a more reliable comparison.

## Full 201-strategy portfolio

The original 21 hand-authored bots are preserved. In addition, `poker_bots.py`
generates 180 named equity-driven variants across 18 strategy families, for
201 distinct built-in classes total. Each generated bot has its own parameter
profile. The generator verifies each batch of ten has unique profiles, and
tests compare their decisions over a broad equity sweep.

List the portfolio or run every strategy against every other strategy:

```bash
python3 run_tournament.py --list-strategies
python3 run_tournament.py --all-strategies --hands 10 --seeds 42,43,44
```

The 201-bot portfolio has 20,100 pairings per seed, so the all-strategy command
defaults to one hand per pairing unless `--hands` is supplied. `--seeds`
repeats the complete round robin on independent shuffled deal sequences and
aggregates matchup wins/losses and chip results. For a multi-seed final in the
four-generation experiment, use `python3 run_evolution.py --hands 100
--final-seeds 42,43,44`.

## Add team bots

Implement an `act(observation)` method that returns an `Action`. The observation
contains five private cards, the public board, pot and stack sizes, call amount,
legal raise bounds, position, and opponent count. Bet/raise amounts are target
street contributions; return `Action.call()` to call the amount due.

```python
from poker_bots import PokerBot
from short_deck_plo import Action, Observation

class TeamBot(PokerBot):
    name = "TeamBot"

    def act(self, observation: Observation) -> Action:
        if observation.call_amount == 0:
            return Action.check()
        return Action.call()
```

Place the bot in an importable Python module, then pass each team as
`--bot NAME=MODULE:CLASS`. When any `--bot` arguments are provided, they replace
the built-in lineup:

```bash
python3 run_tournament.py \
  --bot TeamA=team_a:TeamBot \
  --bot TeamB=team_b:TeamBot \
  --hands 2000 --seed 42
```

`ShortDeckPLOBot` uses sampled heads-up showdown equity, pot odds, and position
to make decisions. The included reference bots are intentionally simple
benchmarks, not optimized opponents. Improve the main bot in `poker_bots.py`,
then compare it against the broad field with the same seeded round-robin
harness. Use `--bot` to include multiple submissions in the same round robin.

## Best strategy for every starting hand

`hand_strategy_search.py` measures which built-in strategy earns the most chips
with each possible starting hand and saves the winner per hand. `super_bot.py`
(next section) reads that table to choose a strategy at the start of every hand.

> **Do not start a full run casually.** The default run is about 724 million
> poker hands and is meant for a large multi-core machine. Use `estimate`
> (plays nothing) or `run --limit N` (a short trial) when working on the code.

### What the search does, exactly

For every starting hand `H`:

1. Every **hero** strategy (default: all 201 built-ins) is given `H`.
2. Each hero plays heads-up against every strategy in the **opponent field**.
   The default field is `lineup`, the 9 bots in the default
   `run_tournament.py` lineup, not all 201. Use `--opponents all` for the full
   field, which takes about 22 times longer.
3. Each **seed** (default `1-5`) gives `--deals-per-seed` deals (default 2).
   A deal is a random opponent hand plus a random five-card board, drawn from
   the 31 cards not in `H`. The shuffle for each deal is seeded from
   `blake2b(f"{seed}|{hand}|{deal}")`, so it is reproducible and different for
   every hand.
4. Each deal is played **twice**, once with the hero on the button and once
   out of position.
5. **Common random numbers.** Every hero plays exactly the same deals against
   the same opponents. Bot randomness is reseeded identically before every
   hand: hero `deal_seed + button`, opponent `deal_seed + button + 1_000_003`.
   A hero's results therefore do not depend on which other heroes are in the
   run (a test checks this).
6. Stacks reset every hand (default 1,000 chips, blinds 5/10).

**Ranking metric.** The ranking uses the hero's **average net chips per hand,
in big blinds**, over all opponents, seeds, deals and both seats. It is *not*
a count of hands or matches won. Chips decide matches in this engine, and a
win count would favour strategies that win many small pots and lose a few big
ones. The strategy with the highest average is stored as `best`; ties are
broken by name.

Hands per hero per starting hand = `opponents × seeds × deals_per_seed × 2`
(default 9 × 5 × 2 × 2 = 180).

**Suit symmetry.** Strategies only care about suits relative to each other, so
hands that are suit relabellings of each other share one result. The 376,992
raw hands collapse to **19,998** classes. A class is represented by its
smallest relabelling (`canonical_hand()`), and `combos` records how many raw
hands it stands for (1 to 24). `--all-combos` tests every raw hand instead,
which is about 19 times more work.

**Noise.** Ten random deals per hand is a small sample. The 180 generated
variants differ only slightly, so the top spot among them is often luck.
`clear_winner` is true when `best` leads `runner_up` by more than
`2 × sqrt(se_best² + se_runner_up²)`. That test is conservative, because the
deals are shared. Add seeds to get more clear winners.

### Commands

```bash
# Count the work without playing any hands (add --ms-per-hand X for a time estimate)
python3 hand_strategy_search.py estimate --seeds 1-10

# Short trial to measure speed: 20 hands, then check the timing it prints
python3 hand_strategy_search.py run --out trial --seeds 1-10 --limit 20

# Full run on every CPU core (--workers N to limit); rerun the same command to resume
python3 hand_strategy_search.py run --out results --seeds 1-10

# Or split across machines: shard K of N on each, then copy all shard files into one folder
python3 hand_strategy_search.py run --out results --seeds 1-10 --shard 0/4

# Build the tables from the shard files, then query them
python3 hand_strategy_search.py merge --out results
python3 hand_strategy_search.py lookup --out results "As Kh Qd Jc Ts" "6s 7s 8h 9h Td"
```

Options shared by `estimate` and `run`:

| Option | Default | Meaning |
|---|---|---|
| `--heroes` | `all` | Strategies compared: `all` (201), `reference` (21 hand-written), `lineup` (9), or comma-separated names from `run_tournament.py --list-strategies` |
| `--opponents` | `lineup` | Field each hero plays against; same choices |
| `--seeds` | `1-5` | `1-10`, `1,2,42`, or a mix |
| `--deals-per-seed` | `2` | Opponent hand and board draws per seed |
| `--stack`, `--small-blind`, `--big-blind` | `1000`, `5`, `10` | Chip settings, reset every hand |
| `--all-combos` | off | Test raw hands instead of suit classes |
| `--shard K/N` | `0/1` | Only hands at index `K, K+N, K+2N…` of the fixed hand order |
| `--limit N` | none | Only the first N hands of this shard |
| `--workers N` | all cores | Worker processes |

`run` also takes `--out` (default `hand_strategy_results`), `--force` and
`--progress-every` (default 25). `merge` and `lookup` take only `--out`.

### Resuming and settings safety

- `run` writes `run_config.json` into `--out` on the first run. A later `run`
  into the same folder with different settings stops with an error, unless
  `--force` is given. Never mix settings in one folder: the averages would not
  be comparable.
- Results go to `shard-KKK-of-NNN.jsonl`, one line per finished hand, flushed
  as each hand finishes. On restart, hands already in the file are skipped. A
  half-written last line from an interruption is ignored, and that hand is
  played again.
- `merge` reads every `shard-*.jsonl` in the folder. If a hand appears twice,
  the first copy wins. It reports how many of the expected hands it found, so
  a missing shard shows up as a count below 19,998.

### Output files (all in `--out`)

| File | Written by | Contents |
|---|---|---|
| `run_config.json` | `run` | The settings: `heroes`, `opponents`, `seeds`, `deals_per_seed`, `starting_stack`, `small_blind`, `big_blind`, `all_combos` |
| `shard-*.jsonl` | `run` | Raw results, one JSON object per hand (format below) |
| `best_strategy_by_hand.json` | `merge` | The table `SuperBot` reads |
| `best_strategy_by_hand.csv` | `merge` | The same table for spreadsheets |
| `strategy_summary.csv` | `merge` | Per strategy: `hands_where_best`, `share_of_all_combos` (weighted by `combos`), `avg_bb_per_hand_all_hands` |

Shard line, with the hand written high card first, ranks `6-9 T J Q K A`, suits `s h d c`:

```json
{"hand": "As Kh Qd Jc Ts", "weight": 24,
 "stats": {"ShortDeckPLO": [180, 12.5, 940.25], "...": []}}
```

`stats[name]` is `[hands played, sum of bb results, sum of squared bb results]`,
enough to rebuild the mean and standard error.

`best_strategy_by_hand.json`:

```json
{"config": {"...": "run_config.json contents"},
 "suit_symmetry": true,
 "hands": {
   "As Kh Qd Jc Ts": {
     "best": "TightValue03", "best_bb_per_hand": 0.42, "best_std_error": 0.11,
     "runner_up": "EquityValue07", "runner_up_bb_per_hand": 0.35, "margin": 0.07,
     "clear_winner": false, "hands_per_strategy": 180, "combos": 24}}}
```

Keys in `hands` are canonical hands. Look up a hand with
`best_strategy_for(hole, load_best_table(folder))`, which canonicalizes it
first, rather than formatting a key yourself.

### Python API (`hand_strategy_search.py`)

- `starting_hands(all_combos=False)`: list of `(hand, combos)` in the fixed order sharding uses.
- `canonical_hand(hand)`, `parse_hand("As Kh Qd Jc Ts")`, `format_hand(hand)`, `hand_key(hand)`.
- `evaluate_hand(hand, SearchConfig(...))`: `{hero: [n, sum_bb, sum_sq_bb]}` for one hand.
- `summarize_hand(stats)`: the per-hand summary dict shown above.
- `load_best_table(path)` and `best_strategy_for(hole, table)`: lookups, `None` for hands not in the table.

## Super bot

`super_bot.SuperBot` is a normal `PokerBot`. On the first decision of each hand
it looks up its five hole cards in `best_strategy_by_hand.json` and passes
every decision for that hand to the stored `best` strategy. Each chosen
strategy keeps one instance for the bot's lifetime.

- **Table location:** the `table_path=` argument, otherwise the
  `SUPER_BOT_TABLE` environment variable, otherwise `./hand_strategy_results`.
  A results folder or the JSON file itself both work. A missing table raises
  `FileNotFoundError` when the bot is created. A table naming a strategy this
  code doesn't have raises `ValueError`.
- **Fallback:** hands not in the table use `fallback=` (default `ShortDeckPLO`).
- **`require_clear_winner=True`:** hands whose `clear_winner` is false also use
  the fallback.
- **New-hand detection:** the engine gives no hand-start signal, so the bot
  treats a hand as new when its hole cards change or the board gets shorter.
- **Inspection:** `current_strategy` is the strategy for the hand in progress.
  `strategy_counts` counts how often each strategy was chosen. `strategy_for(hole)`
  returns the choice without playing.
- **Seeding:** `set_seed()` reseeds the inner strategies, so matches stay reproducible.

Run it in a round robin:

```bash
SUPER_BOT_TABLE=results python3 run_tournament.py \
  --bot Super=super_bot:SuperBot \
  --bot ShortDeckPLO=poker_bots:ShortDeckPLOBot \
  --bot TightAggressive=poker_bots:TightAggressiveBot \
  --hands 2000 --seeds 100,101,102
```

Evaluate it on seeds the search did not use, here 100 and up against a search
over 1-10. Otherwise it is graded on the deals it was tuned on. Note that the
search only sees the `--opponents` field, so the bot is tuned for that field.

### Engine helpers added for this

`short_deck_plo.play_hand(first, second, button=, deck=, ...)` plays one hand
from a prepared deck and returns `(net_seat0, net_seat1)`.
`short_deck_plo.deck_for_deal((hand_seat0, hand_seat1), button, rest)` builds
that deck. `rest` is the other 26 cards: `rest[0:3]` is the flop, `rest[3]`
the turn and `rest[4]` the river. Existing engine behaviour is unchanged.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

`tests/test_hand_strategy_search.py` and `tests/test_super_bot.py` cover dealing,
suit symmetry, the common-deal guarantee, merge and lookup, and SuperBot's
table use, fallback and strategy switching. They use only the cheap
`CallingStation`, `Random` and `Maniac` bots on a few hands, finish in seconds,
and never start the real search.
