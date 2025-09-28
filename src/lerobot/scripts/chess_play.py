#!/usr/bin/env python

from __future__ import annotations

import time
from dataclasses import dataclass
import argparse

try:
    import chess
except Exception:
    chess = None

from lerobot.utils.utils import log_say


@dataclass
class PlayConfig:
    engine: str | None = None  # placeholder; user can integrate engine later


def _parse_args() -> PlayConfig:
    p = argparse.ArgumentParser(description="Play chess with SO-101 (skeleton)")
    p.add_argument("--engine", help="Chess engine (placeholder)")
    a = p.parse_args()
    return PlayConfig(engine=a.engine)


def main(cfg: PlayConfig | None = None) -> None:
    if cfg is None:
        cfg = _parse_args()
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


