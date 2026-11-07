import json
import tempfile
import unittest
from pathlib import Path

from hand_strategy_search import canonical_hand, hand_key, parse_hand
from official_match import OfficialSeat, play_hand
from official_rules import cards_to_official, game
from short_deck_plo import DECK, Observation
from super_bot import SuperBot, load_strategy_table


def _observation(hole, board=(), call_amount=0):
    return Observation(
        seat=0, button=0, hole=hole, board=board, pot=15, stack=995,
        street_contribution=5, current_bet=10, call_amount=call_amount,
        min_raise_to=20, max_raise_to=30, can_raise=True, big_blind=10, opponents=1,
    )


class SuperBotTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.maniac_hand = parse_hand("As Ks Qh Jh Td")
        self.close_hand = parse_hand("6s 7h 8d 9c Ts")
        table = {
            "suit_symmetry": True,
            "hands": {
                hand_key(canonical_hand(self.maniac_hand)): {"best": "Maniac", "clear_winner": True},
                hand_key(canonical_hand(self.close_hand)): {"best": "Maniac", "clear_winner": False},
            },
        }
        Path(self.folder.name, "best_strategy_by_hand.json").write_text(json.dumps(table))
        load_strategy_table.cache_clear()

    def tearDown(self):
        self.folder.cleanup()

    def _bot(self, **options):
        return SuperBot(table_path=self.folder.name, fallback="CallingStation", **options)

    def test_uses_table_strategy_for_any_suit_relabelling(self):
        bot = self._bot()
        relabelled = parse_hand("Ah Kh Qs Js Tc")
        action = bot.act(_observation(relabelled, call_amount=5))
        self.assertEqual(bot.current_strategy, "Maniac")
        self.assertEqual((action.kind, action.amount), ("raise", 30))

    def test_falls_back_for_hands_not_in_table(self):
        bot = self._bot()
        action = bot.act(_observation(parse_hand("6s 6h 7d 8c 9s"), call_amount=5))
        self.assertEqual(bot.current_strategy, "CallingStation")
        self.assertEqual(action.kind, "call")

    def test_can_ignore_hands_without_a_clear_winner(self):
        self.assertEqual(self._bot().strategy_for(self.close_hand), "Maniac")
        bot = self._bot(require_clear_winner=True)
        self.assertEqual(bot.strategy_for(self.close_hand), "CallingStation")
        self.assertEqual(bot.strategy_for(self.maniac_hand), "Maniac")

    def test_switches_strategy_between_hands_in_a_real_hand(self):
        bot, other = OfficialSeat(self._bot()), OfficialSeat(self._bot())
        rest = [card for card in DECK if card not in self.maniac_hand]
        villain, board = rest[:5], rest[5:10]

        def deal(seat0, seat1):
            return game.Deal(hole_cards=(tuple(sorted(cards_to_official(seat0))),
                                         tuple(sorted(cards_to_official(seat1)))),
                             board=cards_to_official(board))

        play_hand((bot, other), deal(self.maniac_hand, villain), button=0)
        self.assertEqual(bot.bot.current_strategy, "Maniac")
        play_hand((bot, other), deal(villain, self.maniac_hand), button=1)
        self.assertEqual(bot.bot.current_strategy, "CallingStation")

    def test_missing_table_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            SuperBot(table_path=str(Path(self.folder.name, "nope")))


if __name__ == "__main__":
    unittest.main()
