"""Play heads-up Short Deck PLO5 against SuperBot in your browser, under the official rules.

    python3 play_vs_superbot.py                         # uses ./hand_strategy_results
    python3 play_vs_superbot.py --table path/to/results
    python3 play_vs_superbot.py --opponent ShortDeckPLO # any built-in bot, no table needed

This opens http://127.0.0.1:8765. It only reads the saved strategy table; it
never imports or runs hand_strategy_search.py.

How it works: every hand is an official game.Game (80-chip stacks, blinds
1/2, fold / check-call / half-pot raise / pot raise, at most 4 raises per
street). Deals come from game.make_deal(session seed, hand number) and the
button alternates, like one half of a tournament match. You are seat 0. The
bot plays through official_match.OfficialSeat. The page polls /api/state and
posts your action to /api/action.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

from official_match import OfficialSeat
from official_rules import evaluator, game
from poker_bots import PokerBot
from run_tournament import ALL_BUILTIN_STRATEGIES

PAGE = Path(__file__).with_name("play_vs_superbot.html")
HUMAN, BOT = 0, 1
STREETS = {game.Street.PREFLOP: "Preflop", game.Street.FLOP: "Flop",
           game.Street.TURN: "Turn", game.Street.RIVER: "River"}
Action = game.Action


def _cards(cards) -> list[str]:
    return [evaluator.card_to_str(card) for card in cards]


def parse_action(kind: str, legal) -> "game.Action":
    """Turn the browser's action name into a legal official Action, or raise ValueError."""
    try:
        action = Action[str(kind)]
    except KeyError:
        raise ValueError(f"Unknown action {kind!r}; use FOLD, CHECK_CALL, RAISE_HALF or "
                         "RAISE_POT.") from None
    if action not in legal:
        names = ", ".join(a.name for a in legal)
        raise ValueError(f"{action.name} is not allowed now; you can {names}.")
    return action


class Table:
    """One player session against one bot. Thread-safe."""

    def __init__(self, opponent_factory: Callable[[], PokerBot], opponent_label: str, *,
                 seed: Optional[str] = None, bot_delay: float = 0.7) -> None:
        self.lock = threading.Lock()
        self.opponent = opponent_factory()
        self.seat = OfficialSeat(self.opponent)
        self.opponent_label = opponent_label
        self.session_seed = str(seed) if seed is not None else secrets.token_hex(8)
        self.seat.set_seed(int.from_bytes(self.session_seed.encode()[:8].ljust(8, b"0"), "big"))
        self.bot_delay = bot_delay
        self.hands_played = 0
        self.session_net = 0
        self.status = "idle"
        self.hand: Optional[game.Game] = None
        self.log: list[dict] = []
        self.result: Optional[dict] = None
        self.error: Optional[str] = None

    # ---- playing -----------------------------------------------------------

    def start_hand(self) -> None:
        with self.lock:
            if self.status not in ("idle", "hand_over"):
                raise ValueError("Finish the current hand first.")
            number = self.hands_played
            self.hand = game.Game(game.make_deal(self.session_seed, number),
                                  game.button_for_deal(number), hand_index=number)
            you_post = game.SMALL_BLIND if self.hand.button_seat == HUMAN else game.BIG_BLIND
            bot_posts = game.BIG_BLIND + game.SMALL_BLIND - you_post
            self.log = [{"street": "Preflop", "seat": -1,
                         "text": f"You post {you_post}, {self.opponent_label} posts {bot_posts}"}]
            self.result = None
            self.error = None
            self.status = "bot_thinking"
        self._continue()

    def submit(self, kind: str) -> None:
        with self.lock:
            if self.status != "your_turn" or self.hand is None:
                raise ValueError("It is not your turn.")
            action = parse_action(kind, self.hand.legal_actions())
            self._apply(HUMAN, action)
            self.status = "bot_thinking"
        self._continue()

    def _continue(self) -> None:
        threading.Thread(target=self._run_bot, daemon=True).start()

    def _run_bot(self) -> None:
        """Let the bot act until it is your turn or the hand ends."""
        while True:
            with self.lock:
                hand = self.hand
                if hand.is_terminal:
                    self._finish()
                    return
                if hand.to_act == HUMAN:
                    self.status = "your_turn"
                    return
                observation = hand.observation(BOT)
            if self.bot_delay:
                time.sleep(self.bot_delay)
            try:
                action = self.seat.act(observation)
            except Exception as error:  # show bot crashes in the page, then check or fold
                action = game.default_action(observation.to_call)
                with self.lock:
                    self.error = f"{self.opponent_label} crashed ({error}); it checks or folds."
            with self.lock:
                self._apply(BOT, action)

    def _apply(self, seat: int, action: "game.Action") -> None:
        hand = self.hand
        facing_bet = hand.observation(seat).to_call > 0 or hand.street == game.Street.PREFLOP
        hand.apply(action)
        record = hand.history[-1]
        who = "You" if seat == HUMAN else self.opponent_label
        s = "" if seat == HUMAN else "s"
        if action == Action.FOLD:
            text = f"{who} fold{s}"
        elif action == Action.CHECK_CALL:
            text = f"{who} call{s} {record.amount}" if record.amount else f"{who} check{s}"
        else:
            verb = ("raise" if facing_bet else "bet") + s
            size = "half pot" if action == Action.RAISE_HALF else "pot"
            text = f"{who} {verb} to {record.street_total} ({size})"
            if hand.stacks[seat] == 0:
                text += ", all-in"
        self.log.append({"street": STREETS[record.street], "seat": seat, "text": text})

    def _finish(self) -> None:
        result = self.hand.result()
        net = result.nets[HUMAN]
        self.hands_played += 1
        self.session_net += net
        if net > 0:
            headline = f"You win {net} chips"
        elif net < 0:
            headline = f"{self.opponent_label} wins {-net} chips"
        else:
            headline = "Split pot"
        last = result.history[-1] if result.history else None
        if result.went_to_showdown:
            how = "at showdown"
        elif last is not None and last.seat == HUMAN:
            how = "- you folded"
        else:
            how = f"- {self.opponent_label} folded"
        showdown = None
        if result.went_to_showdown:
            board = self.hand.deal.board
            scores = [evaluator.omaha_score(self.hand.deal.hole_cards[s], board) for s in (0, 1)]
            showdown = {"you": evaluator.category_name(scores[HUMAN]),
                        "bot": evaluator.category_name(scores[BOT])}
        self.result = {
            "net": net,
            "headline": f"{headline} {how}",
            "showdown": result.went_to_showdown,
            "hands": showdown,
            "bot_hole": _cards(self.hand.deal.hole_cards[BOT]) if result.went_to_showdown else None,
            "strategy": getattr(self.opponent, "current_strategy", None),
            "pot": result.pot,
        }
        self.status = "hand_over"

    # ---- view --------------------------------------------------------------

    def state(self) -> dict:
        with self.lock:
            hand = self.hand
            view = hand.observation(HUMAN) if hand is not None else None
            to_act = None
            if self.status == "your_turn" and view is not None:
                legal = view.legal_actions
                to_act = {
                    "legal": [a.name for a in legal],
                    "call_amount": min(view.to_call, view.my_stack),
                    "raise_half_to": view.raise_to.get(Action.RAISE_HALF),
                    "raise_pot_to": view.raise_to.get(Action.RAISE_POT),
                    "all_in_to": view.my_bet + view.my_stack,
                    "raises_this_street": view.raises_this_street,
                    "raise_cap": game.RAISE_CAP_PER_STREET,
                }
            if hand is None:
                board, pot, stacks, bets = [], 0, [game.STARTING_STACK] * 2, [0, 0]
                button = "you"
            else:
                over = hand.is_terminal
                board = _cards(hand.result().board_revealed if over else view.board)
                pot = hand.pot
                stacks = list(hand.stacks)
                bets = [0, 0] if over else list(hand.bets)
                button = "you" if hand.button_seat == HUMAN else "bot"
                if over:
                    stacks = [game.STARTING_STACK + n for n in hand.result().nets]
            hands = self.hands_played
            return {
                "status": self.status,
                "opponent": self.opponent_label,
                "hand_number": hands + (0 if self.status in ("idle", "hand_over") else 1),
                "button": button,
                "you": {"hole": _cards(view.hole_cards) if view else [], "stack": stacks[HUMAN],
                        "bet": bets[HUMAN]},
                "bot": {"stack": stacks[BOT], "bet": bets[BOT]},
                "board": board,
                "street": STREETS[hand.street] if hand else "Preflop",
                "pot": pot,
                "to_act": to_act,
                "log": list(self.log),
                "result": self.result,
                "error": self.error,
                "session": {
                    "hands": hands,
                    "net": self.session_net,
                    "bb_per_hand": round(self.session_net / game.BIG_BLIND / hands, 2) if hands else 0,
                    "blinds": f"{game.SMALL_BLIND}/{game.BIG_BLIND}",
                    "starting_stack": game.STARTING_STACK,
                },
            }


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
                    table.submit(str(body.get("kind", "")))
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
    parser.add_argument("--seed", help="Seed the deals to replay the same session.")
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
    evaluator.warmup()
    table = Table(factory, label, seed=args.seed, bot_delay=args.bot_delay)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(table))
    url = f"http://{args.host}:{args.port}/"
    print(f"Playing against {label} under the official rules. Open {url}  (Ctrl+C to stop)")
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
