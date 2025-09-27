#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from lerobot.perception.chess.board_model import BoardModel
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import ComplementaryDataProcessorStep


Square = Literal[
    "a1","a2","a3","a4","a5","a6","a7","a8",
    "b1","b2","b3","b4","b5","b6","b7","b8",
    "c1","c2","c3","c4","c5","c6","c7","c8",
    "d1","d2","d3","d4","d5","d6","d7","d8",
    "e1","e2","e3","e4","e5","e6","e7","e8",
    "f1","f2","f3","f4","f5","f6","f7","f8",
    "g1","g2","g3","g4","g5","g6","g7","g8",
    "h1","h2","h3","h4","h5","h6","h7","h8",
]


def square_to_indices(sq: Square) -> tuple[int, int]:
    file_char = sq[0]
    rank_char = sq[1]
    file_idx = ord(file_char) - ord("a")
    rank_idx = int(rank_char) - 1
    return file_idx, rank_idx


@dataclass
class ChessSquareToEEPose(ComplementaryDataProcessorStep):
    board_model: BoardModel
    hover_height_m: float = 0.08
    place_offset_m: float = 0.0

    def complementary_data(self, complementary_data: dict) -> dict:
        move = complementary_data.get("chess_move")  # dict like {"from":"e2","to":"e4"}
        if not move:
            return complementary_data

        src = move["from"]
        dst = move["to"]
        fi_s, ri_s = square_to_indices(src)
        fi_d, ri_d = square_to_indices(dst)

        p_src = self.board_model.square_center_in_board(fi_s, ri_s)
        p_dst = self.board_model.square_center_in_board(fi_d, ri_d)

        Tbb = self.board_model.T_base_board
        if Tbb is None:
            raise ValueError("BoardModel.T_base_board must be set from calibration")

        p_src_base = (Tbb @ np.hstack([p_src, 1.0]))[:3]
        p_dst_base = (Tbb @ np.hstack([p_dst, 1.0]))[:3]

        # Construct hover and touch poses (position-only here; orientation handled later)
        hover_src = np.hstack([p_src_base[:2], [p_src_base[2] + self.hover_height_m]])
        touch_src = p_src_base
        hover_dst = np.hstack([p_dst_base[:2], [p_dst_base[2] + self.hover_height_m]])
        touch_dst = p_dst_base

        out = dict(complementary_data)
        out["ee_waypoint_src_hover_xyz"] = hover_src.astype(float)
        out["ee_waypoint_src_touch_xyz"] = touch_src.astype(float)
        out["ee_waypoint_dst_hover_xyz"] = hover_dst.astype(float)
        out["ee_waypoint_dst_touch_xyz"] = touch_dst.astype(float)
        return out


