#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lerobot.configs.chessboard import ChessboardConfig
from lerobot.perception.chess.board_model import BoardModel
from lerobot.perception.chess.piece_detector import OccupancyDetector
from lerobot.perception.chess.state_builder import build_state_from_occupancy
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import ObservationProcessorStep


@dataclass
class ChessStateFromImage(ObservationProcessorStep):
    camera_key: str
    chess_cfg: ChessboardConfig
    board_model: BoardModel
    detector: OccupancyDetector

    overlay_key: str | None = None

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        frame = observation.get(self.camera_key)
        if frame is None:
            return observation

        occ = self.detector.detect(frame)
        state = build_state_from_occupancy(occ)

        # Attach results in complementary data and optionally add overlay to observation
        comp = self.transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        comp = dict(comp)
        comp["chess_fen"] = state.fen
        comp["chess_occupancy"] = occ.astype(np.uint8)
        self._current_transition[TransitionKey.COMPLEMENTARY_DATA] = comp

        if self.overlay_key:
            observation[self.overlay_key] = frame  # placeholder; later draw grid/labels
        return observation


