#!/usr/bin/env python

from __future__ import annotations

import time
from dataclasses import dataclass

import draccus

try:
    import chess
except Exception:
    chess = None

from lerobot.utils.utils import log_say


@dataclass
class PlayConfig:
    engine: str | None = None  # placeholder; user can integrate engine later


def main(cfg: PlayConfig = draccus.run(PlayConfig)) -> None:
    if chess is None:
        log_say("python-chess not installed. Install it to use chess_play.")
        return
    board = chess.Board()
    log_say("Starting play loop (skeleton). Use chess_move.py for executing moves.")
    while not board.is_game_over():
        time.sleep(1.0)
        break


if __name__ == "__main__":
    main()


