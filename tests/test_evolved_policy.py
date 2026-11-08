import random
import unittest

from evolved_policy import POTSHOVE08, EvolvedPolicy, Genome, mutate, random_genome
from official_match import OfficialSeat, match_record
from official_rules import game
from poker_bots import CallingStationBot, ManiacBot, PotShove08


class EvolvedPolicyTests(unittest.TestCase):
    def test_default_genome_plays_exactly_like_potshove08(self):
        for opponent_class, seed in ((ManiacBot, 3), (CallingStationBot, 4), (PotShove08, 5)):
            mine, original = EvolvedPolicy(POTSHOVE08), OfficialSeat(PotShove08())
            mine.set_seed(seed)
            original.set_seed(seed)
            opponent = OfficialSeat(opponent_class())
            opponent.set_seed(seed + 1)
            decisions = 0
            for hand in range(60):
                table = game.Game(game.make_deal(f"evo{seed}", hand), hand % 2, hand_index=hand)
                while not table.is_terminal:
                    obs = table.observation(table.to_act)
                    if table.to_act == 0:
                        action = mine.act(obs)
                        self.assertEqual(action, original.act(obs))
                        decisions += 1
                    else:
                        action = opponent.act(obs)
                    table.apply(action)
            self.assertGreater(decisions, 60)

    def test_random_and_mutated_genomes_only_play_legal_moves(self):
        rng = random.Random(1)
        genomes = [random_genome(rng) for _ in range(4)] + [mutate(POTSHOVE08, rng) for _ in range(4)]
        for index, genome in enumerate(genomes):
            record = match_record("A", "B", EvolvedPolicy(genome), ManiacBot(), seed=index, deals=15)
            self.assertIn(record["verdict"], ("AGENT1_WIN", "AGENT2_WIN", "CHOP"))

    def test_genome_round_trips_through_a_dict(self):
        genome = random_genome(random.Random(7))
        self.assertEqual(Genome.from_dict(genome.to_dict()), genome)

    def test_adaptation_learns_folds_and_showdown_bluffs(self):
        policy = EvolvedPolicy(Genome(a_bluff=0.5, a_thin=0.3, a_catch=0.5))
        policy.set_seed(1)
        self.assertAlmostEqual(policy.opponent_fold_rate(), 0.40)
        weak_opp = game.Game(game.make_deal("adapt", 0), 0)
        board = weak_opp.deal.board
        for hand in range(10):  # we raised and the opponent folded every time
            policy._hand, policy._we_raised = hand, True
            policy.on_hand_end(game.HandResult(hand, 5, 10, False, (), None, "me"))
        self.assertAlmostEqual(policy.opponent_fold_rate(), (10 + 6) / 25)
        policy._hand, policy._opp_raised = 10, True
        policy.on_hand_end(game.HandResult(10, -20, 40, True, board,
                                           weak_opp.deal.hole_cards[1], "me"))
        self.assertEqual(policy.sd_raises, 1)
        record = match_record("A", "B", policy, ManiacBot(), seed=3, deals=20)
        self.assertIn(record["verdict"], ("AGENT1_WIN", "AGENT2_WIN", "CHOP"))
        self.assertGreater(policy.bet_hands, 0)  # the match runner now calls on_hand_end

    def test_opponent_model_counts_raises(self):
        policy = EvolvedPolicy()
        self.assertAlmostEqual(policy.opponent_aggression(), 0.25)
        policy.opp_actions, policy.opp_raises = 20, 20
        self.assertAlmostEqual(policy.opponent_aggression(), 0.625)
        policy.set_seed(1)
        self.assertAlmostEqual(policy.opponent_aggression(), 0.25)


if __name__ == "__main__":
    unittest.main()
