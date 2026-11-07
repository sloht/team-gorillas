"""Play heads-up Short Deck PLO against SuperBot in your browser.

    python3 play_vs_superbot.py                         # uses ./hand_strategy_results
    python3 play_vs_superbot.py --table path/to/results
    python3 play_vs_superbot.py --opponent ShortDeckPLO # any built-in bot, no table needed

This opens http://127.0.0.1:8765. It only reads the saved strategy table; it
never imports or runs hand_strategy_search.py.

How it works: each hand runs in the real engine (short_deck_plo.play_hand) on
a background thread. Your seat is a HumanBot whose act() waits for the action
you click in the browser. The page polls /api/state and sends actions to
/api/action. Stacks reset every hand and the button alternates, as in
run_match.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

from hand_keys import format_card
from poker_bots import PokerBot
from run_tournament import ALL_BUILTIN_STRATEGIES
from short_deck_plo import DECK, Action, Observation, play_hand

PAGE = Path(__file__).with_name("play_vs_superbot.html")
HUMAN, BOT = 0, 1
STREETS = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}


def _cards(cards) -> list[str]:
    return [format_card(card) for card in cards]


def validate_action(observation: Observation, kind: str, amount: int = 0) -> Action:
    """Turn a browser request into a legal Action or raise ValueError with a reason."""
    call = observation.call_amount
    if kind == "fold":
        if not call:
            raise ValueError("Checking is free, so there is nothing to fold to.")
        return Action.fold()
    if kind == "check":
        if call:
            raise ValueError(f"You face a bet of {call}; call, raise or fold.")
        return Action.check()
    if kind == "call":
        if not call:
            raise ValueError("There is no bet to call; check instead.")
        return Action.call()
    if kind in ("bet", "raise"):
        if not observation.can_raise:
            raise ValueError("You cannot raise here.")
        expected = "raise" if observation.current_bet else "bet"
        if kind != expected:
            raise ValueError(f"There is {'a' if expected == 'raise' else 'no'} bet to face; "
                             f"use {expected}.")
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise ValueError("The amount must be a whole number of chips.")
        lowest = min(observation.min_raise_to, observation.max_raise_to)
        if not lowest <= amount <= observation.max_raise_to:
            raise ValueError(f"{expected.title()} to between {lowest} and "
                             f"{observation.max_raise_to}.")
        return Action.raise_to(amount) if expected == "raise" else Action.bet_to(amount)
    raise ValueError(f"Unknown action {kind!r}.")


class Table:
    """Shared state between the engine thread and the web requests."""

    def __init__(self, opponent_factory: Callable[[], PokerBot], opponent_label: str, *,
                 starting_stack: int = 1_000, small_blind: int = 5, big_blind: int = 10,
                 seed: Optional[int] = None, bot_delay: float = 0.7) -> None:
        self.lock = threading.Lock()
        self.opponent = opponent_factory()
        self.opponent_label = opponent_label
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.bot_delay = bot_delay
        self.rng = random.Random(seed)
        if hasattr(self.opponent, "set_seed"):
            self.opponent.set_seed(self.rng.randrange(2**31))
        self.actions: "queue.Queue[Action]" = queue.Queue()
        self.hands_played = 0
        self.session_net = 0
        self.history: list[int] = []
        self._reset_hand_view()
        self.status = "idle"

    # ---- view kept up to date from every observation ----------------------

    def _reset_hand_view(self) -> None:
        self.log: list[dict] = []
        self.board: tuple = ()
        self.pot = 0
        self.stacks = [self.starting_stack, self.starting_stack]
        self.street_bets = [0, 0]
        self.human_hole: tuple = ()
        self.pending: Optional[Observation] = None
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.button = 0
        self.folded: Optional[int] = None

    def _observe(self, seat: int, observation: Observation) -> None:
        own_total = self.starting_stack - observation.stack
        other_total = observation.pot - own_total
        self.board = observation.board
        self.pot = observation.pot
        self.stacks[seat] = observation.stack
        self.stacks[1 - seat] = self.starting_stack - other_total
        self.street_bets[seat] = observation.street_contribution
        # Whoever is not acting has put in the current bet on this street.
        self.street_bets[1 - seat] = observation.current_bet

    def _record(self, seat: int, observation: Observation, action: Action) -> None:
        who = "You" if seat == HUMAN else self.opponent_label
        if action.kind == "call":
            paid = min(observation.call_amount, observation.stack)
            text = f"{who} call{'s' if seat else ''} {paid}"
        elif action.kind in ("bet", "raise"):
            all_in = action.amount - observation.street_contribution >= observation.stack
            verb = action.kind + ("s" if seat else "")
            text = f"{who} {verb} to {action.amount}" + (" (all-in)" if all_in else "")
        else:
            text = f"{who} {action.kind}{'s' if seat else ''}"
        if action.kind == "fold":
            self.folded = seat
        self.log.append({"street": STREETS[len(observation.board)], "seat": seat, "text": text})

    # ---- the two players ---------------------------------------------------

    @staticmethod
    def _round_already_closed(observation: Observation) -> bool:
        """The engine re-asks a bettor after being called; checking is the only move.

        _betting_round rebuilds the to-act list with _clockwise(seat), which in
        heads-up includes the bettor. No chips can move on that turn, so the UI
        answers it with a check and does not show or log it.
        """
        return (not observation.call_amount and not observation.can_raise
                and observation.current_bet > 0
                and observation.street_contribution == observation.current_bet)

    def human_act(self, observation: Observation) -> Action:
        if self._round_already_closed(observation):
            return Action.check()
        with self.lock:
            self._observe(HUMAN, observation)
            self.pending = observation
            self.status = "your_turn"
        action = self.actions.get()
        with self.lock:
            self._record(HUMAN, observation, action)
            self.pending = None
            self.status = "bot_thinking"
        return action

    def bot_act(self, observation: Observation) -> Action:
        if self._round_already_closed(observation):
            return self.opponent.act(observation)  # still asked, so its state stays as in tournaments
        with self.lock:
            self._observe(BOT, observation)
            self.status = "bot_thinking"
        if self.bot_delay:
            time.sleep(self.bot_delay)
        action = self.opponent.act(observation)
        with self.lock:
            self._record(BOT, observation, action)
        return action

    # ---- hands -------------------------------------------------------------

    def start_hand(self) -> None:
        with self.lock:
            if self.status not in ("idle", "hand_over"):
                raise ValueError("Finish the current hand first.")
            self._reset_hand_view()
            self.button = self.hands_played % 2
            self.status = "dealing"
            deck = list(DECK)
            self.rng.shuffle(deck)
            # Cards are popped from the end, alternating from the button.
            order = deck[::-1]
            holes = {self.button: order[0:10:2], 1 - self.button: order[1:10:2]}
            self.human_hole = tuple(holes[HUMAN])
            self._bot_hole = tuple(holes[BOT])
            self._full_board = tuple(order[10:15])
            while not self.actions.empty():
                self.actions.get_nowait()
        threading.Thread(target=self._play, args=(deck,), daemon=True).start()

    def _play(self, deck: list) -> None:
        human = _Seat(self.human_act, "You")
        bot = _Seat(self.bot_act, self.opponent_label)
        try:
            net = play_hand(human, bot, button=self.button, deck=deck,
                            starting_stack=self.starting_stack,
                            small_blind=self.small_blind, big_blind=self.big_blind)
        except Exception as error:  # surface engine errors in the page
            with self.lock:
                self.error = f"The hand stopped with an error: {error}"
                self.status = "hand_over"
            return
        with self.lock:
            showdown = self.folded is None
            if showdown:
                self.board = self._full_board
            self.stacks = [self.starting_stack + net[HUMAN], self.starting_stack + net[BOT]]
            self.pot = abs(net[HUMAN])  # shown as "changed hands" once the hand is over
            self.street_bets = [0, 0]
            self.hands_played += 1
            self.session_net += net[HUMAN]
            self.history.append(net[HUMAN])
            if net[HUMAN] > 0:
                headline = f"You win {net[HUMAN]} chips"
            elif net[HUMAN] < 0:
                headline = f"{self.opponent_label} wins {-net[HUMAN]} chips"
            else:
                headline = "Split pot"
            how = ("at showdown" if showdown
                   else "- you folded" if self.folded == HUMAN
                   else f"- {self.opponent_label} folded")
            self.result = {
                "net": net[HUMAN],
                "headline": f"{headline} {how}",
                "showdown": showdown,
                "bot_hole": _cards(self._bot_hole) if showdown else None,
                "strategy": getattr(self.opponent, "current_strategy", None),
            }
            self.status = "hand_over"

    def submit(self, kind: str, amount: int = 0) -> None:
        with self.lock:
            if self.status != "your_turn" or self.pending is None:
                raise ValueError("It is not your turn.")
            action = validate_action(self.pending, kind, amount)
            self.status = "bot_thinking"
        self.actions.put(action)

    def state(self) -> dict:
        with self.lock:
            pending = self.pending
            to_act = None
            if pending is not None:
                to_act = {
                    "call_amount": min(pending.call_amount, pending.stack),
                    "current_bet": pending.current_bet,
                    "street_contribution": pending.street_contribution,
                    "can_raise": pending.can_raise,
                    "raise_kind": "raise" if pending.current_bet else "bet",
                    "min_raise_to": min(pending.min_raise_to, pending.max_raise_to),
                    "max_raise_to": pending.max_raise_to,
                    "all_in_to": pending.street_contribution + pending.stack,
                    "pot": pending.pot,
                }
            hands = self.hands_played
            return {
                "status": self.status,
                "opponent": self.opponent_label,
                "hand_number": hands + (0 if self.status in ("idle", "hand_over") else 1),
                "button": "you" if self.button == HUMAN else "bot",
                "you": {"hole": _cards(self.human_hole), "stack": self.stacks[HUMAN],
                        "bet": self.street_bets[HUMAN]},
                "bot": {"stack": self.stacks[BOT], "bet": self.street_bets[BOT]},
                "board": _cards(self.board),
                "street": STREETS[len(self.board)],
                "pot": self.pot,
                "to_act": to_act,
                "log": list(self.log),
                "result": self.result,
                "error": self.error,
                "session": {
                    "hands": hands,
                    "net": self.session_net,
                    "bb_per_hand": round(self.session_net / self.big_blind / hands, 2) if hands else 0,
                    "big_blind": self.big_blind,
                    "blinds": f"{self.small_blind}/{self.big_blind}",
                    "starting_stack": self.starting_stack,
                },
            }


class _Seat:
    """Adapter so the engine can call a Table method as if it were a bot."""

    def __init__(self, act: Callable[[Observation], Action], name: str) -> None:
        self.act = act
        self.name = name


def make_handler(table: Table) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(table.state())
            else:
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/new_hand":
                    table.start_hand()
                elif self.path == "/api/action":
                    table.submit(str(body.get("kind", "")), body.get("amount", 0))
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")
                    return
            except (ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self._json(table.state())

    return Handler


def build_opponent(name: str, table_path: Optional[str]) -> tuple[Callable[[], PokerBot], str]:
    if name == "SuperBot":
        # Imported here so other opponents do not need a strategy table at all.
        from super_bot import SuperBot, load_strategy_table
        load_strategy_table(str(table_path or _default_table()))  # fail early if missing
        return (lambda: SuperBot(table_path=table_path)), "SuperBot"
    strategies = dict(ALL_BUILTIN_STRATEGIES)
    if name not in strategies:
        raise SystemExit(f"Unknown opponent {name!r}. Use SuperBot or a name from "
                         "run_tournament.py --list-strategies.")
    return strategies[name], name


def _default_table() -> str:
    from super_bot import DEFAULT_TABLE, TABLE_ENV
    return os.environ.get(TABLE_ENV, DEFAULT_TABLE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", default="SuperBot",
                        help="SuperBot (default) or any built-in strategy name.")
    parser.add_argument("--table", help="SuperBot's results folder or best_strategy_by_hand.json. "
                                        "Default: $SUPER_BOT_TABLE or ./hand_strategy_results.")
    parser.add_argument("--stack", type=int, default=1_000)
    parser.add_argument("--small-blind", type=int, default=5)
    parser.add_argument("--big-blind", type=int, default=10)
    parser.add_argument("--seed", type=int, help="Seed the shuffles to replay the same session.")
    parser.add_argument("--bot-delay", type=float, default=0.7,
                        help="Seconds the bot pauses before acting, so you can follow along.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab.")
    args = parser.parse_args()

    try:
        factory, label = build_opponent(args.opponent, args.table)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"{error}\n\nTo try the table now without a strategy table, run:\n"
                         "  python3 play_vs_superbot.py --opponent ShortDeckPLO")
    table = Table(factory, label, starting_stack=args.stack, small_blind=args.small_blind,
                  big_blind=args.big_blind, seed=args.seed, bot_delay=args.bot_delay)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(table))
    url = f"http://{args.host}:{args.port}/"
    print(f"Playing against {label}. Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
