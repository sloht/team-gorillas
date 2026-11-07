import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from hand_strategy_search import (
    SearchConfig,
    load_qualified,
    main,
    rank_qualifiers,
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
from official_rules import game
from short_deck_plo import Action


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
                  seeds=(1, 2), deals_per_seed=2, all_combos=False)
    values.update(overrides)
    return SearchConfig(**values)


class DealingTests(unittest.TestCase):
    def test_hero_always_holds_the_hand_under_official_rules(self):
        hand = parse_hand("As Kh Qd Jc Ts")
        recorder, opponent = _RecordingBot(), _RecordingBot()
        config = _config(heroes=("Recorder",), opponents=("Opponent",))
        seen_holes, seen_boards = set(), set()
        original_act = recorder.act

        def act(observation):
            seen_holes.add(tuple(sorted(observation.hole)))
            if len(observation.board) == 5:
                seen_boards.add(observation.board)
            self.assertEqual(observation.big_blind, game.BIG_BLIND)
            self.assertLessEqual(observation.stack, game.STARTING_STACK)
            return original_act(observation)

        recorder.act = act
        stats = evaluate_hand(hand, config, {"Recorder": recorder}, {"Opponent": opponent})
        self.assertEqual(seen_holes, {tuple(sorted(hand))})
        self.assertEqual(len(seen_boards), 4)  # 2 seeds x 2 deals, each played from both seats
        self.assertEqual(stats["Recorder"][0], 2 * 2 * 2)


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


class QualifyTests(unittest.TestCase):
    def test_rank_by_points_then_bb_per_deal(self):
        # A beats B, B beats C, C beats A: all on 1 point, so bb per deal decides.
        # A: (0.8 - 0.2) / 2 = +0.30, C: (-0.3 + 0.2) / 2 = -0.05, B: (-0.8 + 0.3) / 2 = -0.25
        matches = [
            {"left": "A", "right": "B", "seed": 1, "points": [1, 0], "mean_bb": 0.8},
            {"left": "B", "right": "C", "seed": 1, "points": [1, 0], "mean_bb": 0.3},
            {"left": "A", "right": "C", "seed": 1, "points": [0, 1], "mean_bb": -0.2},
        ]
        ranking = rank_qualifiers(matches, ["A", "B", "C"])
        self.assertEqual([row["name"] for row in ranking], ["A", "C", "B"])
        self.assertEqual([row["points"] for row in ranking], [1, 1, 1])
        self.assertAlmostEqual(ranking[0]["avg_bb_per_deal"], 0.3)

        # A chop is worth half a point and outranks a loss.
        chop = [{"left": "A", "right": "B", "seed": 1, "points": [0.5, 0.5], "mean_bb": 0.0}]
        self.assertEqual(rank_qualifiers(chop, ["A", "B"])[0]["chops"], 1)

    def test_qualify_writes_top_n_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as out, \
                redirect_stderr(io.StringIO()) as log:
            args = ["qualify", "--out", folder, "--pool", "CallingStation,Random,Maniac",
                    "--top", "2", "--seeds", "1,2", "--deals", "2", "--workers", "1"]
            main(args)
            data = json.loads(Path(folder, "qualifiers.json").read_text())
            self.assertEqual(len(data["qualified"]), 2)
            self.assertEqual(len(data["ranking"]), 3)
            self.assertEqual(sum(row["matches"] for row in data["ranking"]), 3 * 2 * 2)
            self.assertEqual(load_qualified(folder), tuple(data["qualified"]))

            # Saved: a second qualify reads the file and plays nothing.
            main(args)
            self.assertIn("Already qualified", out.getvalue())
            self.assertNotIn("matches already done", log.getvalue().split("Qualifying:")[-1])

            # Lost qualifiers.json but kept the match log: rebuilt without replaying.
            Path(folder, "qualifiers.json").unlink()
            main(args)
            self.assertIn("6 matches already done, 0 to go", log.getvalue())
            self.assertEqual(load_qualified(folder), tuple(data["qualified"]))

            # A copied qualifiers.json can be used from anywhere with --qualifiers.
            with tempfile.TemporaryDirectory() as other:
                copy = Path(other, "top.json")
                copy.write_text(Path(folder, "qualifiers.json").read_text())
                main(["estimate", "--out", other, "--qualifiers", str(copy)])
                self.assertIn(f"Hero strategies        : 2", out.getvalue())

    def test_run_needs_qualifiers_unless_heroes_given(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                main(["run", "--out", folder, "--limit", "1", "--workers", "1"])
            self.assertIn("qualify", str(stopped.exception))
            main(["estimate", "--out", folder])  # estimate works before qualifying


if __name__ == "__main__":
    unittest.main()
