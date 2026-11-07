import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from official_rules import game
from play_vs_superbot import Table, build_opponent, parse_action
from poker_bots import CallingStationBot, ManiacBot

ROOT = Path(__file__).resolve().parents[1]
A = game.Action


def _wait(table, statuses, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = table.state()
        if state["status"] in statuses:
            return state
        time.sleep(0.005)
    raise AssertionError(f"Table stuck in {table.state()['status']}")


def _play_out(table, choose):
    """Play one hand, choosing each action with choose(state['to_act'])."""
    table.start_hand()
    while True:
        state = _wait(table, {"your_turn", "hand_over"})
        if state["status"] == "hand_over":
            return state
        table.submit(choose(state["to_act"]))


class ParseActionTests(unittest.TestCase):
    def test_accepts_only_legal_official_actions(self):
        legal = (A.FOLD, A.CHECK_CALL, A.RAISE_HALF, A.RAISE_POT)
        self.assertEqual(parse_action("RAISE_POT", legal), A.RAISE_POT)
        with self.assertRaises(ValueError):
            parse_action("FOLD", (A.CHECK_CALL,))  # folding is not allowed when checking is free
        with self.assertRaises(ValueError):
            parse_action("raise", legal)
        with self.assertRaises(ValueError):
            parse_action("RAISE_HALF", (A.FOLD, A.CHECK_CALL))  # raise cap reached


class TableTests(unittest.TestCase):
    def _table(self, bot=CallingStationBot, seed="t"):
        return Table(bot, "Bot", seed=seed, bot_delay=0)

    def test_plays_official_hands_and_keeps_score(self):
        table = self._table()
        total = 0
        for _ in range(6):
            state = _play_out(table, lambda t: "CHECK_CALL")
            self.assertTrue(state["result"]["showdown"])
            self.assertEqual(len(state["result"]["bot_hole"]), 5)
            self.assertEqual(len(state["board"]), 5)
            self.assertEqual(state["you"]["stack"] + state["bot"]["stack"],
                             2 * game.STARTING_STACK)
            total += state["result"]["net"]
        self.assertEqual(state["session"]["hands"], 6)
        self.assertEqual(state["session"]["net"], total)
        self.assertEqual(state["session"]["blinds"], "1/2")

    def test_deals_match_the_official_dealer_and_button(self):
        table = self._table(seed="replay")
        buttons = []
        for number in range(2):
            table.start_hand()
            state = _wait(table, {"your_turn", "hand_over"})
            deal = game.make_deal("replay", number)
            self.assertEqual(state["you"]["hole"], [game.card_to_str(c) for c in deal.hole_cards[0]])
            buttons.append(state["button"])
            while state["status"] == "your_turn":
                table.submit("FOLD" if "FOLD" in state["to_act"]["legal"] else "CHECK_CALL")
                state = _wait(table, {"your_turn", "hand_over"})
            if not state["result"]["showdown"]:
                self.assertIsNone(state["result"]["bot_hole"])  # folded hands stay hidden
        self.assertEqual(buttons, ["you", "bot"])

    def test_raise_buttons_show_the_official_amounts(self):
        table = self._table(bot=ManiacBot)
        table.start_hand()
        state = _wait(table, {"your_turn"})
        to_act = state["to_act"]
        observation = table.hand.observation(0)
        self.assertEqual(to_act["raise_half_to"], observation.raise_to.get(A.RAISE_HALF))
        self.assertEqual(to_act["raise_pot_to"], observation.raise_to.get(A.RAISE_POT))
        self.assertEqual(set(to_act["legal"]), {a.name for a in observation.legal_actions})

    def test_raise_cap_and_all_in_against_a_maniac(self):
        table = self._table(bot=ManiacBot)
        for _ in range(4):
            state = _play_out(table, lambda t: "RAISE_POT" if "RAISE_POT" in t["legal"]
                              else "CHECK_CALL")
            self.assertIn(abs(state["result"]["net"]), (0, game.STARTING_STACK))
            self.assertTrue(any("all-in" in entry["text"] for entry in state["log"]))

    def test_rejects_actions_out_of_turn_and_double_deals(self):
        table = self._table()
        with self.assertRaises(ValueError):
            table.submit("CHECK_CALL")
        table.start_hand()
        _wait(table, {"your_turn"})
        with self.assertRaises(ValueError):
            table.start_hand()
        with self.assertRaises(ValueError):
            table.submit("SHOVE")
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
