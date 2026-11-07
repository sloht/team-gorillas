import json
import tempfile
import unittest
from pathlib import Path

from hand_strategy_search import (
    SearchConfig,
    best_strategy_for,
    canonical_hand,
    command_merge,
    evaluate_hand,
    format_hand,
    hand_key,
    load_best_table,
    parse_hand,
    parse_seeds,
    starting_hands,
    summarize_hand,
)
from poker_bots import PokerBot
from short_deck_plo import DECK, Action, deck_for_deal, play_hand


class _RecordingBot(PokerBot):
    name = "Recorder"

    def __init__(self) -> None:
        super().__init__()
        self.hole = None
        self.boards = []

    def act(self, observation):
        self.hole = observation.hole
        self.boards.append(observation.board)
        return Action.call() if observation.call_amount else Action.check()


def _config(**overrides):
    values = dict(heroes=("CallingStation", "Random"), opponents=("CallingStation",),
                  seeds=(1, 2), deals_per_seed=2, starting_stack=1_000,
                  small_blind=5, big_blind=10, all_combos=False)
    values.update(overrides)
    return SearchConfig(**values)


class DealingTests(unittest.TestCase):
    def test_deck_for_deal_gives_each_seat_its_hand_and_board(self):
        first, second = DECK[:5], DECK[5:10]
        rest = list(DECK[10:])
        for button in (0, 1):
            bots = (_RecordingBot(), _RecordingBot())
            deck = deck_for_deal((first, second), button, rest)
            net = play_hand(*bots, button=button, deck=deck)
            self.assertEqual(sorted(bots[0].hole), sorted(first))
            self.assertEqual(sorted(bots[1].hole), sorted(second))
            self.assertEqual(bots[0].boards[-1], tuple(rest[:5]))
            self.assertEqual(sum(net), 0)

    def test_deck_for_deal_rejects_overlapping_cards(self):
        with self.assertRaises(ValueError):
            deck_for_deal((DECK[:5], DECK[:5]), 0, list(DECK[10:]))


class StartingHandTests(unittest.TestCase):
    def test_suit_distinct_hands_cover_every_combo(self):
        hands = starting_hands()
        self.assertEqual(len(hands), 19_998)
        self.assertEqual(sum(weight for _, weight in hands), 376_992)

    def test_canonical_hand_ignores_suit_labels(self):
        hand = parse_hand("As Ks Qh Jd 9c")
        relabelled = parse_hand("Ah Kh Qc Js 9d")
        self.assertEqual(canonical_hand(hand), canonical_hand(relabelled))
        self.assertNotEqual(canonical_hand(hand), canonical_hand(parse_hand("As Ks Qs Jd 9c")))

    def test_parse_and_format_round_trip(self):
        self.assertEqual(format_hand(parse_hand("6c,Ts,As,9h,Kd")), "As Kd Ts 9h 6c")
        with self.assertRaises(ValueError):
            parse_hand("As As Kd Qd Jd")
        with self.assertRaises(ValueError):
            parse_hand("5s Ks Qd Jd Td")

    def test_parse_seeds(self):
        self.assertEqual(parse_seeds("1-3,42"), (1, 2, 3, 42))
        self.assertEqual(parse_seeds("-2,5"), (-2, 5))


class EvaluateHandTests(unittest.TestCase):
    def test_counts_and_common_deals(self):
        hand = starting_hands()[0][0]
        config = _config(heroes=("CallingStation",), opponents=("CallingStation", "Random"))
        stats = evaluate_hand(hand, config)
        self.assertEqual(stats["CallingStation"][0], 2 * 2 * 2 * 2)
        self.assertEqual(stats, evaluate_hand(hand, config))

    def test_each_hero_sees_the_same_deals_regardless_of_lineup(self):
        hand = parse_hand("As Ks Qh Jh Td")
        alone = evaluate_hand(hand, _config(heroes=("CallingStation",)))
        together = evaluate_hand(hand, _config(heroes=("Random", "CallingStation")))
        self.assertEqual(alone["CallingStation"], together["CallingStation"])


class MergeAndLookupTests(unittest.TestCase):
    def test_summarize_picks_highest_mean(self):
        summary = summarize_hand({"A": [100, 50.0, 300.0], "B": [100, -20.0, 300.0]})
        self.assertEqual(summary["best"], "A")
        self.assertEqual(summary["runner_up"], "B")
        self.assertAlmostEqual(summary["margin"], 0.7)

    def test_merge_then_lookup(self):
        hand = parse_hand("As Kh Qd Jc Ts")
        key = hand_key(canonical_hand(hand))
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "run_config.json").write_text(json.dumps(_config().comparable()))
            record = {"hand": key, "weight": 24,
                      "stats": {"CallingStation": [8, -4.0, 10.0], "Random": [8, 6.0, 12.0]}}
            (out / "shard-000-of-001.jsonl").write_text(json.dumps(record) + "\n{broken")

            class Args:
                pass

            args = Args()
            args.out = str(out)
            command_merge(args)
            table = load_best_table(out)
            result = best_strategy_for("Ah Ks Qc Jd Th", table)
            self.assertEqual(result["best"], "Random")
            self.assertEqual(result["combos"], 24)
            self.assertIsNone(best_strategy_for("6s 7s 8s 9s Ts", table))


if __name__ == "__main__":
    unittest.main()
