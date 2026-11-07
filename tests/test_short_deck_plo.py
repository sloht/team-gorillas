import unittest
from unittest.mock import patch

from poker_bots import (
    AdaptiveCallBot,
    BlockerBluffBot,
    BoardTextureBot,
    BluffBot,
    CallingStationBot,
    DelayedAggressorBot,
    FieldBalancedBot,
    GENERATED_STRATEGY_CLASSES,
    LooseAggressiveBot,
    ManiacBot,
    PotOddsBot,
    PotControlBot,
    PositionPressureBot,
    RandomBot,
    RangePressureBot,
    RockBot,
    SPRShoveBot,
    ShortDeckPLOBot,
    SmallBallBot,
    TightAggressiveBot,
    TrapValueBot,
    ValueFirstBot,
    estimate_equity,
)
from run_evolution import ALL_COMPETITORS, GENERATIONS, ORIGINAL, advance_field
from run_tournament import ALL_BUILTIN_STRATEGIES
from short_deck_plo import (
    Action,
    DECK,
    Observation,
    Standing,
    omaha_rank,
    rank_five,
    run_match,
    run_multi_seed_round_robin,
)


class HandRankingTests(unittest.TestCase):
    def test_short_deck_deck_contains_36_cards(self):
        self.assertEqual(len(DECK), 36)
        self.assertEqual({rank for rank, _ in DECK}, set(range(6, 15)))

    def test_flush_beats_full_house(self):
        flush = [(6, 0), (8, 0), (10, 0), (12, 0), (14, 0)]
        full_house = [(6, 0), (6, 1), (6, 2), (9, 0), (9, 1)]
        self.assertGreater(rank_five(flush), rank_five(full_house))

    def test_short_deck_wheel_is_six_to_ten(self):
        wheel = [(14, 0), (6, 1), (7, 2), (8, 3), (9, 0)]
        self.assertEqual(rank_five(wheel), (4, 9))

    def test_omaha_must_use_exactly_two_hole_cards(self):
        hole = [(6, 0), (7, 1), (8, 2), (6, 3), (7, 0)]
        board = [(10, 3), (11, 3), (12, 3), (13, 3), (14, 3)]
        self.assertLess(omaha_rank(hole, board)[0], 8)

    def test_omaha_rejects_wrong_hole_card_count(self):
        with self.assertRaises(ValueError):
            omaha_rank([(6, 0)] * 4, [(7, 0), (8, 0), (9, 0)])


class BotAndMatchTests(unittest.TestCase):
    def test_equity_estimate_is_reproducible_and_bounded(self):
        observation = Observation(
            seat=0, button=0,
            hole=((6, 0), (7, 0), (8, 1), (9, 1), (14, 2)),
            board=((10, 0), (11, 0), (12, 2)),
            pot=30, stack=970, street_contribution=0, current_bet=0,
            call_amount=0, min_raise_to=10, max_raise_to=30,
            can_raise=True, big_blind=10, opponents=1,
        )
        first = estimate_equity(observation, samples=8)
        second = estimate_equity(observation, samples=8)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 1.0)

    def test_same_seed_replays_same_match(self):
        def play():
            return run_match(
                ShortDeckPLOBot(), RandomBot(), hands=12, seed=123,
            )

        self.assertEqual(play(), play())

    def test_calling_stations_complete_heads_up_hands(self):
        result = run_match(
            CallingStationBot(), CallingStationBot(), hands=4, seed=3,
        )
        self.assertEqual(result.hands_played, 4)
        self.assertEqual(sum(result.net_chips), 0)

    def test_multi_seed_round_robin_aggregates_records(self):
        standings = run_multi_seed_round_robin(
            [("CallA", CallingStationBot), ("CallB", CallingStationBot)],
            seeds=(3, 5),
            hands_per_match=2,
        )
        self.assertEqual(len(standings), 2)
        self.assertTrue(all(row.matches == 2 for row in standings))
        self.assertTrue(all(row.seed_match_counts == (1, 1) for row in standings))
        self.assertTrue(all(row.seed_win_rate_range is not None for row in standings))
        self.assertEqual(sum(row.net_chips for row in standings), 0)

    def test_action_helpers_return_expected_kinds(self):
        self.assertEqual(Action.bet_to(20).kind, "bet")
        self.assertEqual(Action.raise_to(30).amount, 30)

    def test_four_generations_add_three_strategies_each(self):
        self.assertEqual(len(GENERATIONS), 4)
        self.assertTrue(all(len(generation) == 3 for generation in GENERATIONS))
        self.assertEqual(len(ALL_COMPETITORS), 21)
        self.assertEqual(len({name for name, _ in ALL_COMPETITORS}), 21)

    def test_strategy_portfolio_has_201_unique_names_and_classes(self):
        names = [name for name, _ in ALL_BUILTIN_STRATEGIES]
        classes = [bot_class for _, bot_class in ALL_BUILTIN_STRATEGIES]
        self.assertEqual(len(names), 201)
        self.assertEqual(len(set(names)), 201)
        self.assertEqual(len(set(classes)), 201)
        self.assertEqual(len(GENERATED_STRATEGY_CLASSES), 180)
        self.assertEqual(
            len({bot_class.profile for bot_class in GENERATED_STRATEGY_CLASSES}),
            180,
        )

    def test_each_batch_of_ten_has_unique_profiles_and_actions(self):
        equities = [step / 100 for step in range(101)]
        for start in range(0, len(GENERATED_STRATEGY_CLASSES), 10):
            batch = GENERATED_STRATEGY_CLASSES[start:start + 10]
            with self.subTest(batch=batch[0].name):
                self.assertEqual(len({bot_class.profile for bot_class in batch}), 10)
                observations = [
                    Observation(
                        seat=0, button=button,
                        hole=((6, 0), (7, 0), (8, 1), (9, 1), (14, 2)),
                        board=((10, 0), (11, 0), (12, 2)),
                        pot=30, stack=970, street_contribution=0, current_bet=0,
                        call_amount=0, min_raise_to=10, max_raise_to=30,
                        can_raise=True, big_blind=10, opponents=1,
                    )
                    for button in (0, 1)
                ]
                observations.extend(
                    Observation(
                        seat=0, button=button,
                        hole=((6, 0), (7, 0), (8, 1), (9, 1), (14, 2)),
                        board=((10, 0), (11, 0), (12, 2)),
                        pot=50, stack=970, street_contribution=0, current_bet=20,
                        call_amount=20, min_raise_to=40, max_raise_to=100,
                        can_raise=True, big_blind=10, opponents=1,
                    )
                    for button in (0, 1)
                )
                signatures = []
                for bot_class in batch:
                    bot = bot_class()
                    bot.set_seed(123)
                    signature = []
                    for observation in observations:
                        with patch("poker_bots.estimate_equity", side_effect=equities):
                            actions = [
                                bot.act(observation)
                                for _ in equities
                            ]
                        signature.extend((action.kind, action.amount) for action in actions)
                    signatures.append(tuple(signature))
                self.assertEqual(
                    len(set(signatures)), 10,
                    "Strategies in each group of ten must produce distinct decisions "
                    "over a broad equity sweep.",
                )

    def test_elimination_keeps_top_six_then_adds_three(self):
        standings = [
            Standing(name, 8, wins, 8 - wins, 0, wins * 100, 0)
            for wins, (name, _) in enumerate(ORIGINAL)
        ]
        next_field = advance_field(ORIGINAL, standings, GENERATIONS[0])
        self.assertEqual(len(next_field), 9)
        self.assertEqual(
            {name for name, _ in next_field},
            {name for name, _ in ORIGINAL[-6:]} | {name for name, _ in GENERATIONS[0]},
        )

    def test_every_new_strategy_can_complete_a_match(self):
        strategies = [
            ValueFirstBot, PotOddsBot, PositionPressureBot, SPRShoveBot,
            TrapValueBot, SmallBallBot, BoardTextureBot, RangePressureBot,
            AdaptiveCallBot, DelayedAggressorBot, BlockerBluffBot, FieldBalancedBot,
        ]
        for strategy in strategies:
            with self.subTest(strategy=strategy.name):
                result = run_match(strategy(), CallingStationBot(), hands=2, seed=7)
                self.assertEqual(result.hands_played, 2)


if __name__ == "__main__":
    unittest.main()
