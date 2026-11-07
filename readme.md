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

## Tests

```bash
python3 -m unittest discover -s tests -v
```
