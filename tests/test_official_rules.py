import hashlib
import re
import unittest
from pathlib import Path

from official_match import (
    OfficialSeat, play_half, play_hand, run_match, to_bot_observation, to_official_action,
)
from official_rules import OFFICIAL_DIR, cards_from_official, game, judge_engine
from poker_bots import CallingStationBot, RandomBot, TightAggressiveBot
from run_tournament import ALL_BUILTIN_STRATEGIES
from short_deck_plo import Action as BotAction

A = game.Action


class OfficialFilesTests(unittest.TestCase):
    def test_official_files_are_unedited_copies(self):
        listed = dict(re.findall(r"\| `(\w+\.py)` \| `([0-9a-f]{64})` \|",
                                 (OFFICIAL_DIR / "SOURCE.md").read_text()))
        self.assertEqual(set(listed), {"game.py", "evaluator.py", "judge_engine.py"})
        for name, digest in listed.items():
            with self.subTest(file=name):
                actual = hashlib.sha256((OFFICIAL_DIR / name).read_bytes()).hexdigest()
                self.assertEqual(actual, digest, f"official/{name} was edited")


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.hand = game.Game(game.make_deal("adapter", 0), button_seat=0)

    def test_observation_carries_the_official_numbers(self):
        official = self.hand.observation(0)  # button, preflop, facing the big blind
        bot_view = to_bot_observation(official)
        self.assertEqual(bot_view.hole, cards_from_official(official.hole_cards))
        self.assertEqual(bot_view.pot, game.SMALL_BLIND + game.BIG_BLIND)
        self.assertEqual(bot_view.call_amount, game.BIG_BLIND - game.SMALL_BLIND)
        self.assertEqual(bot_view.current_bet, game.BIG_BLIND)
        self.assertEqual(bot_view.min_raise_to, official.raise_to[A.RAISE_HALF])
        self.assertEqual(bot_view.max_raise_to, official.raise_to[A.RAISE_POT])
        self.assertTrue(bot_view.can_raise)
        self.assertTrue(bot_view.in_position)
        self.assertTrue(bot_view.preflop)
        self.assertFalse(to_bot_observation(self.hand.observation(1)).in_position)

    def test_bet_sizes_map_to_the_nearer_official_raise(self):
        official = self.hand.observation(0)
        half, pot = official.raise_to[A.RAISE_HALF], official.raise_to[A.RAISE_POT]
        self.assertEqual(to_official_action(BotAction.raise_to(half), official), A.RAISE_HALF)
        self.assertEqual(to_official_action(BotAction.raise_to(pot), official), A.RAISE_POT)
        self.assertEqual(to_official_action(BotAction.raise_to(10**6), official), A.RAISE_POT)
        self.assertEqual(to_official_action(BotAction.raise_to((half + pot) // 2), official),
                         A.RAISE_HALF if (half + pot) // 2 - half <= pot - (half + pot) // 2
                         else A.RAISE_POT)
        self.assertEqual(to_official_action(BotAction.fold(), official), A.FOLD)
        self.assertEqual(to_official_action(BotAction.call(), official), A.CHECK_CALL)

    def test_illegal_bot_moves_become_check_or_call(self):
        self.hand.apply(A.CHECK_CALL)  # button limps; big blind may check for free
        official = self.hand.observation(1)
        self.assertNotIn(A.FOLD, official.legal_actions)
        self.assertEqual(to_official_action(BotAction.fold(), official), A.CHECK_CALL)
        capped = game.Game(game.make_deal("cap", 0), button_seat=0)
        for _ in range(game.RAISE_CAP_PER_STREET):
            if A.RAISE_HALF not in capped.legal_actions():
                break
            capped.apply(A.RAISE_HALF)
        official = capped.observation(capped.to_act)
        if A.RAISE_HALF not in official.legal_actions:
            self.assertEqual(to_official_action(BotAction.raise_to(50), official), A.CHECK_CALL)
            self.assertFalse(to_bot_observation(official).can_raise)

    def test_every_strategy_only_makes_legal_official_moves(self):
        opponent = OfficialSeat(RandomBot(seed=1))
        for name, factory in ALL_BUILTIN_STRATEGIES:
            with self.subTest(strategy=name):
                seat = OfficialSeat(factory())
                for deal_number in range(2):
                    # Game.apply raises IllegalActionError on any illegal move.
                    play_hand((seat, opponent), game.make_deal("legal", deal_number),
                              game.button_for_deal(deal_number))


class DuplicateScoringTests(unittest.TestCase):
    def test_run_match_is_the_judges_combine_halves(self):
        first, second = TightAggressiveBot(), CallingStationBot()
        result = run_match(first, second, deals=30, seed="judge")
        half1 = play_half(TightAggressiveBot(), CallingStationBot(), seed="judge", half=1, deals=30)
        half2 = play_half(TightAggressiveBot(), CallingStationBot(), seed="judge", half=2, deals=30)
        expected = judge_engine.combine_halves(half1, half2)
        self.assertEqual(result.to_dict(include_timing=False),
                         expected.to_dict(include_timing=False))
        self.assertEqual(result.verdict, judge_engine.decide_verdict(*result.ci)[0])

    def test_halves_swap_seats_on_the_same_cards(self):
        half1 = play_half(CallingStationBot(), CallingStationBot(), seed="s", half=1, deals=6)
        half2 = play_half(CallingStationBot(), CallingStationBot(), seed="s", half=2, deals=6)
        self.assertEqual((half1.agent1_seat, half2.agent1_seat), (0, 1))
        # Identical players: whatever seat 0 won in half 1, agent1 loses holding seat 1's cards.
        self.assertEqual([a + b for a, b in zip(half1.nets_agent1, half2.nets_agent1)], [0] * 6)

    def test_swapping_deterministic_players_flips_the_score(self):
        ab = run_match(TightAggressiveBot(), CallingStationBot(), deals=40, seed=5)
        ba = run_match(CallingStationBot(), TightAggressiveBot(), deals=40, seed=5)
        self.assertAlmostEqual(ab.mean_bb, -ba.mean_bb)


if __name__ == "__main__":
    unittest.main()
