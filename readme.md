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

It runs in two stages:

1. **`qualify`**, run once: a round robin of all 201 strategies keeps the
   **top 40**. The list is saved to `qualifiers.json` and never needs to be
   recomputed.
2. **`run`**, the per-hand search: compares only those 40 strategies for every
   starting hand.

> **Do not start a full run casually.** Qualifying is about 3 million poker
> hands and the default per-hand search about 144 million. Both are meant for a
> large multi-core machine. Use `estimate`, `qualify --estimate-only` (neither
> plays anything) or `run --limit N` (a short trial) when working on the code.

### Stage 1: the qualifying tournament (top 40)

`qualify` plays every pair of strategies in the pool (default: all 201, so
20,100 pairings) with `run_match`. Each pairing is played on every seed
(default `1001-1003`, kept apart from the search seeds) for `--hands` hands
(default 50). Strategies are ranked exactly like `run_tournament.py`: by the
number of matches won (one match is one pairing on one seed), then by total
net chips, then by name. The top `--top` (default 40) qualify.

**Stored so it never runs again.** `qualify` writes:

- `qualifiers.json`: `{"config": {...}, "top": 40, "qualified": [40 names, best first], "ranking": [every strategy's matches, match_wins, match_losses, match_ties, net_chips]}`. This is the file `run` reads.
- `qualifier_matches.jsonl`: one line per finished match, `{"left", "right", "seed", "net": [left, right]}`.
- `qualifier_config.json`: the settings, used to refuse a resume with different settings unless `--force`.

Running `qualify` again with the same settings and `qualifiers.json` present
prints the saved list and plays nothing. Delete `qualifiers.json` to rebuild
it: matches already in `qualifier_matches.jsonl` are not replayed. An
interrupted `qualify` resumes the same way. To use the list somewhere else,
for example a copy committed to the repo or sent to another machine, pass
`run --qualifiers path/to/qualifiers.json`.

### Stage 2: what the per-hand search does, exactly

For every starting hand `H`:

1. Every **hero** strategy is given `H`. The default heroes are `qualified`:
   the 40 names in `qualifiers.json`. `run` stops with instructions if that
   file is missing. `--heroes all|reference|lineup|NAME,…` bypasses qualifying.
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

**Noise.** Ten random deals per hand is a small sample. Some of the 40
qualifiers may be near-identical generated variants, so the top spot between
them is often luck.
`clear_winner` is true when `best` leads `runner_up` by more than
`2 × sqrt(se_best² + se_runner_up²)`. That test is conservative, because the
deals are shared. Add seeds to get more clear winners.

### Commands

```bash
# Stage 1, once: qualifying round robin, saves results/qualifiers.json (resumable)
python3 hand_strategy_search.py qualify --out results --estimate-only   # size only
python3 hand_strategy_search.py qualify --out results

# Count the per-hand work without playing (add --ms-per-hand X for a time estimate)
python3 hand_strategy_search.py estimate --out results --seeds 1-10

# Short trial to measure speed: 20 hands, then check the timing it prints
python3 hand_strategy_search.py run --out trial --seeds 1-10 --limit 20

# Full run on every CPU core (--workers N to limit); rerun the same command to resume
python3 hand_strategy_search.py run --out results --seeds 1-10

# Or split across machines: copy qualifiers.json to each, run shard K of N on each,
# then copy all shard files into one folder
python3 hand_strategy_search.py run --out results --seeds 1-10 --shard 0/4

# Build the tables from the shard files, then query them
python3 hand_strategy_search.py merge --out results
python3 hand_strategy_search.py lookup --out results "As Kh Qd Jc Ts" "6s 7s 8h 9h Td"
```

Options shared by `estimate` and `run`:

| Option | Default | Meaning |
|---|---|---|
| `--heroes` | `qualified` | Strategies compared: `qualified` (the saved top 40), `all` (201), `reference` (21 hand-written), `lineup` (9), or comma-separated names from `run_tournament.py --list-strategies` |
| `--out` | `hand_strategy_results` | Results folder; also where `qualifiers.json` is read from |
| `--qualifiers` | `<out>/qualifiers.json` | Read the top strategies from another saved file |
| `--opponents` | `lineup` | Field each hero plays against; same choices |
| `--seeds` | `1-5` | `1-10`, `1,2,42`, or a mix |
| `--deals-per-seed` | `2` | Opponent hand and board draws per seed |
| `--stack`, `--small-blind`, `--big-blind` | `1000`, `5`, `10` | Chip settings, reset every hand |
| `--all-combos` | off | Test raw hands instead of suit classes |
| `--shard K/N` | `0/1` | Only hands at index `K, K+N, K+2N…` of the fixed hand order |
| `--limit N` | none | Only the first N hands of this shard |
| `--workers N` | all cores | Worker processes |

`run` also takes `--force` and `--progress-every` (default 25). `merge` and
`lookup` take only `--out`. `qualify` takes `--pool` (default `all`), `--top`
(40), `--seeds` (`1001-1003`), `--hands` (50), `--stack`, `--small-blind`,
`--big-blind`, `--out`, `--workers`, `--force`, `--estimate-only` and
`--progress-every`.

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

### Shortening the run if it takes too long

Total work is simple to predict:

- Qualifying hands = `pool × (pool − 1) / 2 × qualify seeds × --hands`
- Search hands = `starting hands × heroes × opponents × seeds × deals_per_seed × 2`

Every factor is a setting. Check any combination first with
`qualify --estimate-only` and `estimate` (neither plays a hand). Time a short
trial with `run --limit 20` to get milliseconds per hand, then pass that to
`estimate --ms-per-hand X --workers N` for a run time. The levers, roughly in
order of how much they save for what they cost:

| Lever | Setting | Effect on work | Cost to quality |
|---|---|---|---|
| Fewer heroes | `qualify --top 20` (or 10) | Search scales linearly: 40 → 20 halves it | SuperBot chooses from fewer strategies |
| Fewer opponents | `--opponents ShortDeckPLO,TightAggressive,LooseAggressive` | Linear: 9 → 3 is ⅓ | Bot tuned to a narrower field |
| Fewer seeds | `run --seeds 1-3` | Linear: 5 → 3 is 60% | More noise, fewer `clear_winner` hands |
| One deal per seed | `--deals-per-seed 1` | Halves the search | More noise |
| Smaller qualifying pool | `qualify --pool reference` (21) or a name list | Pairings fall from 20,100 to 210 | Generated variants never compete |
| Shorter qualifying matches | `qualify --hands 20`, `--seeds 1001` | Linear | Noisier top-40 ranking |
| Skip qualifying | `run --heroes reference` or `--heroes NAME,NAME,…` | No stage 1 at all | You choose the heroes yourself |
| Reuse a saved list | `run --qualifiers saved/qualifiers.json` | No stage 1 at all | None, if the saved list suits you |
| Keep suit symmetry | the default; avoid `--all-combos` | `--all-combos` is ~19× more | None |
| More machines | `--shard K/N` on N machines | Same total, ~1/N of the wall time | None |
| All cores | the default; `--workers N` only to limit | Wall time scales with cores | None |
| Partial results | stop any time, then `merge` | Use what has finished | Missing hands use SuperBot's fallback |

Example totals, from the estimate commands:

| Settings | Qualifying hands | Search hands |
|---|---|---|
| Defaults (top 40, 9 opponents, seeds 1-5, 2 deals) | 3,015,000 | 143,985,600 |
| `run --seeds 1-3` | 3,015,000 | 86,391,360 |
| `run --deals-per-seed 1` | 3,015,000 | 71,992,800 |
| `qualify --top 20 --hands 20`; `run --seeds 1-3 --deals-per-seed 1` | 1,206,000 | 21,597,840 |
| `qualify --pool reference --top 10 --seeds 1001 --hands 30`; `run --opponents ShortDeckPLO,TightAggressive,LooseAggressive --seeds 1-3 --deals-per-seed 1` | 6,300 | 3,599,640 |
| Defaults with `--all-combos` | 3,015,000 | 2,714,342,400 |

`merge` works on whatever shard lines exist, and SuperBot uses its fallback
for any hand not yet in the table. So a run can be stopped early and still
produce a usable SuperBot, and a later `run` with the same settings fills in
the rest.

Options that need care or code changes, none of them tested here:

- **PyPy.** The code is pure Python with no third-party packages (`rl_common.py`
  is not used by the search), so `pypy3 hand_strategy_search.py …` should run
  unchanged. Pure-Python loops like these often run several times faster
  under PyPy. Time a `--limit 20` trial with both interpreters before
  committing to it.
- **Faster equity estimates.** Most of the time goes to `estimate_equity`
  in `poker_bots.py`, a Monte Carlo loop over `omaha_rank`. Reducing each
  strategy's `equity_samples` changes how the strategies play, so the results
  would describe different bots; don't do it just for speed. A faster
  `omaha_rank` (for example, a precomputed rank lookup table) keeps the
  results identical and is the best code-level speed-up. Check it against the
  current function on many random hands.
- **GPU.** The code does not use a GPU, and the betting logic would not gain
  from one: each hand is a branching chain of decisions. Only the equity
  sampling could be moved to a GPU, by rewriting `omaha_rank` and
  `estimate_equity` as batched array code (for example in PyTorch or CuPy).
  That is a large rewrite, and the equity values must match the current ones
  or every strategy's behaviour shifts. Try PyPy, more cores and the settings
  above first.

### Output files (all in `--out`)

| File | Written by | Contents |
|---|---|---|
| `qualifiers.json` | `qualify` | The saved top 40 (`qualified`) and the full ranking |
| `qualifier_matches.jsonl` | `qualify` | Every finished qualifying match, for resuming |
| `qualifier_config.json` | `qualify` | Qualifying settings |
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
- `rank_qualifiers(matches, pool)`: the qualifying ranking. `load_qualified(out, path=None)` reads the saved top strategies.

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
- **Choices:** the table only names strategies the search compared, so with
  the default settings SuperBot only ever picks from the qualified top 40.
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

### Keeping playing separate from the search

`super_bot.py` and `play_vs_superbot.py` read only the saved
`best_strategy_by_hand.json`. They never import `hand_strategy_search.py`, so
loading or playing SuperBot cannot start, or even load, the simulation. The
card-text and hand-key helpers both sides need live in `hand_keys.py`, a small
module with no simulation code. `hand_strategy_search.py` re-exports them for
older callers. A test (`test_playing_never_loads_the_strategy_search`) fails if
the play path ever imports the search module.

## Play against SuperBot

`play_vs_superbot.py` is a local web app for playing heads-up against SuperBot
in your browser. It needs nothing beyond the standard library.

```bash
python3 play_vs_superbot.py                          # SuperBot, table from $SUPER_BOT_TABLE or ./hand_strategy_results
python3 play_vs_superbot.py --table results          # point at a results folder or the JSON file
python3 play_vs_superbot.py --opponent ShortDeckPLO  # any built-in bot; no table needed
```

It opens `http://127.0.0.1:8765`. The page shows your five cards, the board,
the pot, both stacks and street bets, and the dealer button. Controls are Fold,
Check or Call, and Bet or Raise with a slider, a chip box, and Min, Half pot
and Pot (or All-in) sizes. Keyboard: F fold, C check or call, R bet or raise,
N next hand. The side log lists every action by street.

At a showdown, SuperBot's cards are revealed. After each hand the page names
the strategy SuperBot played that hand with. The header keeps a running score
in chips and big blinds per hand. Stacks reset every hand and the button
alternates, as in `run_match`.

Options: `--stack`, `--small-blind`, `--big-blind`, `--seed` (replay the same
shuffles), `--bot-delay` (seconds the bot pauses so you can follow along,
default 0.7), `--host`, `--port`, `--no-browser`. Without a strategy table,
`--opponent SuperBot` stops with a message explaining how to make one or play
another bot.

**How it works, for agents changing it.** Each hand runs in the real engine
(`play_hand`) on a background thread. Your seat is a `_Seat` whose `act()` is
`Table.human_act`. That method publishes the observation and blocks on a queue
until the browser posts an action. The browser polls `GET /api/state`, which
returns JSON with status, cards as text such as `"As"`, stacks, `to_act` with
the legal amounts, the log and the result. It sends `POST /api/action
{"kind": "fold|check|call|bet|raise", "amount": N}` and `POST /api/new_hand`.
`validate_action()` checks every action against the engine's rules before it
reaches the engine, so an illegal click returns HTTP 400 with a reason instead
of crashing the hand. The page is `play_vs_superbot.html` (plain HTML, CSS and
JS), served from the same folder.

**Engine quirk this works around.** In `short_deck_plo._Hand._betting_round`,
after a bet or raise the to-act list is rebuilt with `_clockwise(seat)`. In
heads-up that list includes the bettor, so after being called the bettor is
asked to act once more. Its only legal move then is a check, so no chips
move. The UI answers that extra turn with a check without showing or logging
it. The engine itself is unchanged, so tournaments and the strategy search
still make that extra `act()` call. It does not change chip results, but it
does consume one extra random draw for bots like `RandomBot`.

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
table use, fallback and strategy switching. `tests/test_play_vs_superbot.py`
covers action checking, full hands played through the web app's `Table`,
scoring, button rotation, the hidden extra check, and the guarantee that
playing never imports the search. They use only the cheap
`CallingStation`, `Random` and `Maniac` bots on a few hands, finish in seconds,
and never start the real search.
