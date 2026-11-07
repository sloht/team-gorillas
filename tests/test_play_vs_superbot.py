import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from play_vs_superbot import Table, build_opponent, validate_action
from poker_bots import CallingStationBot
from short_deck_plo import DECK, Observation

ROOT = Path(__file__).resolve().parents[1]


def _observation(call_amount=5, current_bet=10, can_raise=True, min_to=20, max_to=30, stack=995):
    return Observation(
        seat=0, button=0, hole=DECK[:5], board=(), pot=15, stack=stack,
        street_contribution=current_bet - call_amount, current_bet=current_bet,
        call_amount=call_amount, min_raise_to=min_to, max_raise_to=max_to,
        can_raise=can_raise, big_blind=10, opponents=1,
    )


def _wait(table, statuses, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = table.state()
        if state["status"] in statuses:
            return state
        time.sleep(0.005)
    raise AssertionError(f"Table stuck in {table.state()['status']}")


class ValidateActionTests(unittest.TestCase):
    def test_legal_actions(self):
        obs = _observation()
        self.assertEqual(validate_action(obs, "fold").kind, "fold")
        self.assertEqual(validate_action(obs, "call").kind, "call")
        self.assertEqual(validate_action(obs, "raise", 25).amount, 25)

    def test_illegal_actions_explain_why(self):
        obs = _observation()
        for kind, amount in (("check", 0), ("raise", 19), ("raise", 31), ("bet", 25),
                             ("raise", "25"), ("dance", 0)):
            with self.assertRaises(ValueError):
                validate_action(obs, kind, amount)
        free = _observation(call_amount=0, current_bet=0, min_to=10, max_to=15)
        with self.assertRaises(ValueError):
            validate_action(free, "fold")
        with self.assertRaises(ValueError):
            validate_action(free, "call")
        self.assertEqual(validate_action(free, "bet", 10).kind, "bet")

    def test_short_all_in_below_minimum_is_allowed(self):
        obs = _observation(min_to=40, max_to=25)
        self.assertEqual(validate_action(obs, "raise", 25).amount, 25)


class TableTests(unittest.TestCase):
    def _table(self):
        return Table(CallingStationBot, "Station", seed=7, bot_delay=0)

    def test_plays_full_hands_and_keeps_score(self):
        table = self._table()
        total = 0
        for _ in range(6):
            table.start_hand()
            while True:
                state = _wait(table, {"your_turn", "hand_over"})
                if state["status"] == "hand_over":
                    break
                to_act = state["to_act"]
                table.submit("call" if to_act["call_amount"] else "check")
            self.assertTrue(state["result"]["showdown"])
            self.assertEqual(len(state["result"]["bot_hole"]), 5)
            self.assertEqual(len(state["board"]), 5)
            self.assertEqual(state["you"]["stack"] + state["bot"]["stack"], 2_000)
            total += state["result"]["net"]
        self.assertEqual(state["session"]["hands"], 6)
        self.assertEqual(state["session"]["net"], total)

    def test_fold_hides_bot_cards_and_alternates_button(self):
        table = self._table()
        buttons = []
        for _ in range(2):
            table.start_hand()
            state = _wait(table, {"your_turn"})
            buttons.append(state["button"])
            if state["to_act"]["call_amount"]:
                table.submit("fold")
            else:  # big blind with a free check: bet, the station calls, then fold later
                table.submit("check")
            state = _wait(table, {"your_turn", "hand_over"})
            while state["status"] == "your_turn":
                table.submit("check" if not state["to_act"]["call_amount"] else "fold")
                state = _wait(table, {"your_turn", "hand_over"})
        self.assertEqual(buttons, ["you", "bot"])

    def test_never_asks_or_logs_a_check_after_a_called_bet(self):
        from poker_bots import ManiacBot
        table = Table(ManiacBot, "Maniac", seed=3, bot_delay=0)
        for _ in range(4):
            table.start_hand()
            while True:
                state = _wait(table, {"your_turn", "hand_over"})
                if state["status"] == "hand_over":
                    break
                to_act = state["to_act"]
                self.assertTrue(to_act["call_amount"] or to_act["can_raise"]
                                or to_act["current_bet"] == 0)
                table.submit("call" if to_act["call_amount"] else "check")
            log = state["log"]
            for before, after in zip(log, log[1:]):
                # A call that matches a bet ends the street (except preflop limps).
                if (" call" in before["text"] and before["street"] != "Preflop"
                        and after["street"] == before["street"]):
                    self.fail(f"Action logged after a closing call: {log}")

    def test_rejects_actions_out_of_turn_and_double_deals(self):
        table = self._table()
        with self.assertRaises(ValueError):
            table.submit("call")
        table.start_hand()
        _wait(table, {"your_turn"})
        with self.assertRaises(ValueError):
            table.start_hand()
        with self.assertRaises(ValueError):
            table.submit("raise", 10**6)
        self.assertEqual(table.state()["status"], "your_turn")


class OpponentLoadingTests(unittest.TestCase):
    def test_superbot_needs_a_table_and_other_bots_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                build_opponent("SuperBot", folder)
            factory, label = build_opponent("ShortDeckPLO", folder)
            self.assertEqual(label, "ShortDeckPLO")
            Path(folder, "best_strategy_by_hand.json").write_text(
                json.dumps({"suit_symmetry": True, "hands": {}}))
            factory, label = build_opponent("SuperBot", folder)
            self.assertEqual(factory().fallback, "ShortDeckPLO")

    def test_playing_never_loads_the_strategy_search(self):
        code = ("import sys, play_vs_superbot, super_bot; "
                "print('hand_strategy_search' in sys.modules)")
        output = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout
        self.assertEqual(output.strip(), "False")


if __name__ == "__main__":
    unittest.main()
