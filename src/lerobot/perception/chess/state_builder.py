#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChessState:
    occupancy: np.ndarray  # (8,8) bool
    fen: str


def occupancy_to_fen(occ: np.ndarray) -> str:
    """Convert boolean occupancy to a pseudo-FEN using 'P' for any piece and '1' for empty runs.

    This is a placeholder to get board-state strings quickly. Later we can integrate python-chess
    for color and piece types.
    """
    if occ.shape != (8, 8):
        raise ValueError("occupancy must be (8,8)")
    # Ranks 8..1 top->bottom; our occ is [row, col] with row 0 as rank 1 -> invert row order
    parts = []
    for r in range(7, -1, -1):
        run = 0
        row_parts = []
        for c in range(8):
            if not occ[r, c]:
                run += 1
            else:
                if run > 0:
                    row_parts.append(str(run))
                    run = 0
                row_parts.append("P")
        if run > 0:
            row_parts.append(str(run))
        parts.append("".join(row_parts) or "8")
    # FEN requires side to move, castling, en-passant, halfmove, fullmove; provide defaults
    board_part = "/".join(parts)
    suffix = " w - - 0 1"
    return board_part + suffix


def build_state_from_occupancy(occ: np.ndarray) -> ChessState:
    fen = occupancy_to_fen(occ)
    return ChessState(occupancy=occ.astype(bool), fen=fen)


