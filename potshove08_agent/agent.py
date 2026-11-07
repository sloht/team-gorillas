"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import random
import time
import traceback
from functools import lru_cache
from hashlib import blake2b

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import SUIT_CHARS, card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Datathon26"
AGENT_NAME = "PotShove08"


# ===== YOUR CODE HERE =====
#
# PotShove08, the strongest strategy in the team's qualifying round robin,
# rewritten to run on its own. It makes exactly the decisions of
# poker_bots.PotShove08 played through official_match.OfficialSeat in the team
# repo:
#   * equity: 14 Monte Carlo samples of one opponent hand plus the rest of
#     the board, scored with the official omaha_score. Samples are drawn in
#     the team repo's card order and seeded from the cards, so the numbers
#     match the original exactly.
#   * thresholds: the PotShove08 profile below (fold, call, value-raise and
#     occasional bluff, adjusted for pot odds, position, street and board
#     texture).
#   * sizing: it always aims for the pot-sized raise, which becomes RAISE_POT
#     (or RAISE_HALF in the rare spot where both raises total the same).

# PotShove08's parameters (poker_bots._make_profile("PotShove", variant 8)).
CALL_MARGIN = 0.055
CALL_FLOOR = 0.29750000000000004
RAISE_THRESHOLD = 0.6950000000000001
RAISE_WHEN_CALLED = 0.7625
BLUFF_THRESHOLD = 0.3925
BLUFF_RATE = 0.1
BET_FRACTION = 1.0
POSITION_EDGE = 0.038
TEXTURE_SENSITIVITY = 0.081
PREFLOP_ADJUSTMENT = 0.031
LATE_STREET_ADJUSTMENT = 0.015
EQUITY_SAMPLES = 14

# Time safety valve. The judge gives 8 s of thinking per 1000-hand half; when
# it runs out the bot can only check or fold for the rest of the half. A full
# half normally uses 2-4 s here, so this never triggers at normal speed. On a
# much slower machine it switches to fewer equity samples (play stops matching
# PotShove08 exactly, but stays close) instead of running out of time.
SLOW_AFTER_SECONDS, SLOW_SAMPLES = 5.0, 6
VERY_SLOW_AFTER_SECONDS, VERY_SLOW_SAMPLES = 6.5, 2

# The team repo's card format: (rank 6..14, suit 0..3 in "shdc" order), and
# its deck order. Only used so the equity samples match the original bit for bit.
_TEAM_SUITS = "shdc"
_TEAM_DECK = tuple((rank, suit) for suit in range(4) for rank in range(6, 15))


def _to_team(card: int) -> tuple:
    return (card >> 2) + 6, _TEAM_SUITS.index(SUIT_CHARS[card & 3])


def _to_official(card: tuple) -> int:
    rank, suit = card
    return ((rank - 6) << 2) | SUIT_CHARS.index(_TEAM_SUITS[suit])


@lru_cache(maxsize=65_536)
def _equity(hole: tuple, board: tuple, opponents: int = 1, samples: int = EQUITY_SAMPLES) -> float:
    """Share of sampled showdowns won (ties split) against one random hand."""
    state = repr((hole, board, opponents, samples)).encode("ascii")
    rng = random.Random(int.from_bytes(blake2b(state, digest_size=8).digest(), "big"))
    known = set(hole) | set(board)
    remaining = [card for card in _TEAM_DECK if card not in known]
    missing_board = 5 - len(board)
    draw_count = missing_board + opponents * 5
    official_hole = tuple(_to_official(card) for card in hole)
    official_board = tuple(_to_official(card) for card in board)
    official_remaining = {card: _to_official(card) for card in remaining}
    equity = 0.0
    for _ in range(samples):
        draw = [official_remaining[card] for card in rng.sample(remaining, draw_count)]
        final_board = (*official_board, *draw[:missing_board])
        hero = omaha_score(official_hole, final_board)
        opponent_ranks = []
        cursor = missing_board
        for _ in range(opponents):
            opponent_ranks.append(omaha_score(draw[cursor:cursor + 5], final_board))
            cursor += 5
        best = max((hero, *opponent_ranks))
        if hero == best:
            equity += 1.0 / (1 + sum(rank == best for rank in opponent_ranks))
    return equity / samples


def _texture_danger(board: tuple) -> float:
    """How coordinated the board is: paired, three of a suit, ranks close together."""
    ranks = [rank for rank, _ in board]
    suits = [suit for _, suit in board]
    danger = 0.0
    if len(set(ranks)) < len(ranks):
        danger += 0.8
    if len(suits) >= 3 and max(suits.count(suit) for suit in set(suits)) >= 3:
        danger += 1.0
    if len(ranks) >= 3 and max(ranks) - min(ranks) <= 4:
        danger += 0.6
    return danger


def _raise(obs: Observation) -> Action:
    """Aim for BET_FRACTION of the way to the pot raise; play the nearer official raise."""
    current_bet = max(obs.my_bet, obs.opp_bet)
    minimum = obs.raise_to[Action.RAISE_HALF]
    maximum = obs.raise_to[Action.RAISE_POT]
    if maximum < minimum:
        target = maximum
    else:
        target = round(current_bet + (maximum - current_bet) * BET_FRACTION)
        target = min(maximum, max(minimum, target))
    return Action.RAISE_HALF if abs(target - minimum) <= abs(target - maximum) else Action.RAISE_POT


def potshove08(obs: Observation, rng: random.Random, samples: int = EQUITY_SAMPLES) -> Action:
    """PotShove08's decision for this observation (always one of obs.legal_actions)."""
    hole = tuple(_to_team(card) for card in obs.hole_cards)
    board = tuple(_to_team(card) for card in obs.board)
    can_raise = Action.RAISE_HALF in obs.legal_actions and Action.RAISE_POT in obs.legal_actions
    equity = _equity(hole, board, 1, samples)
    texture = _texture_danger(board) * TEXTURE_SENSITIVITY
    position = POSITION_EDGE if obs.is_button else -POSITION_EDGE
    street = (PREFLOP_ADJUSTMENT if not board
              else LATE_STREET_ADJUSTMENT if len(board) >= 4 else 0.0)

    if obs.to_call:
        pot_odds = obs.to_call / (obs.pot + obs.to_call)
        call_threshold = max(CALL_FLOOR, pot_odds + CALL_MARGIN + texture + street - position)
        if equity < call_threshold:
            return Action.FOLD if Action.FOLD in obs.legal_actions else Action.CHECK_CALL
        if can_raise and equity >= RAISE_WHEN_CALLED + texture:
            return _raise(obs)
        return Action.CHECK_CALL

    value_bet = equity >= RAISE_THRESHOLD + texture + street - position
    # Same short-circuit as the original: the bluff draw only happens past the threshold.
    bluff = equity >= BLUFF_THRESHOLD + texture and rng.random() < BLUFF_RATE
    if can_raise and (value_bet or bluff):
        return _raise(obs)
    return Action.CHECK_CALL


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands).

    `memory` is a fresh, empty dict every half. Store anything you want to
    remember between hands in it, e.g. memory["opp_raises"] = 0.
    `config` holds the match settings (starting_stack, big_blind, ...).

    In the tournament, each half runs in a brand-new copy of your server,
    so nothing you learn in one half carries over to the other half.
    """
    memory["rng"] = random.Random()  # only used for the occasional bluff
    memory["think_seconds"] = 0.0    # time spent in choose_action this half
    memory["hands_played"] = 0


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    What you can see in obs:
        hand_index          which hand of the half this is (0, 1, 2, ...)
        street              Street.PREFLOP, Street.FLOP, Street.TURN or Street.RIVER
        hole_cards          your 5 cards as ints, sorted ascending (0-35)
        board               board (community) cards seen so far (0, 3, 4 or 5 ints)
        pot                 total chips both players put in this hand
        to_call             chips you must add to match the opponent
        my_bet, opp_bet     chips you,opponent put in on THIS street
        my_stack, opp_stack chips each player still has behind
        is_button           True if you are the button (small blind)
        legal_actions       the Actions you may return right now
        raise_to            {RAISE_HALF: total, RAISE_POT: total} for legal raises
        min_raise_to        smallest normal raise total (all-ins may be lower)
        raises_this_street  raises so far this street (max 4)
        action_history      this hand's actions so far; each has
                            street, is_me, action, amount, street_total
        big_blind           size of the big blind in chips

    Actions: Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
    Cards are ints: card_rank(c) is 0 (six) .. 8 (ace), card_suit(c) is 0..3,
    and cards_to_str(obs.hole_cards) prints them like "6h 7c 9s Kd Ah".
    Once all 5 board cards are out, omaha_score(obs.hole_cards, obs.board)
    scores your hand (higher is better).
    """

    start = time.perf_counter()
    used = memory["think_seconds"]
    samples = (EQUITY_SAMPLES if used < SLOW_AFTER_SECONDS
               else SLOW_SAMPLES if used < VERY_SLOW_AFTER_SECONDS else VERY_SLOW_SAMPLES)
    action = potshove08(obs, memory["rng"], samples)
    memory["think_seconds"] = used + time.perf_counter() - start
    # Belt and braces: never send an illegal action (each one costs an error).
    return action if action in obs.legal_actions else Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Optional: learn from what happened.

    To save time, the judge sends finished hands along with your next
    decision request, so this always runs BEFORE choose_action() sees the
    next hand. (The last hands of a half arrive at the end of the half.)

    result has: hand_index, my_net (chips won or lost), pot,
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
    only after a showdown, else None), winner ('me', 'opp' or 'split').
    """
    memory["hands_played"] += 1

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
