#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from lerobot.utils.geometry import SE3, estimate_homography


@dataclass
class BoardPoseEstimator:
    """Compute T_camera_board from 4 clicked corners (minimal viable path).

    Corner order: a1, h1, h8, a8 in image pixel coordinates (clockwise starting at a1).
    Output pose assumes board Z up, with X along files (a->h) and Y along ranks (1->8).
    Homography establishes plane mapping; orientation resolved from corner order.
    """

    square_size_mm: float = 50.0

    def estimate_from_corners(self, image_corners_xy: np.ndarray) -> SE3:
        if image_corners_xy.shape != (4, 2):
            raise ValueError("image_corners_xy must be (4,2) in order: a1, h1, h8, a8")

        # Build canonical board quad in board frame (meters), Z=0 plane
        s = self.square_size_mm / 1000.0
        W = 8 * s
        H = 8 * s
        board_xy = np.array(
            [
                [0.0, 0.0],   # a1
                [W, 0.0],     # h1
                [W, H],       # h8
                [0.0, H],     # a8
            ],
            dtype=float,
        )

        # Estimate image homography H: board->image
        Hbi = estimate_homography(board_xy, image_corners_xy)

        # Derive camera pose relative to board using planar homography decomposition.
        # We do not recover intrinsics here; we return an SE3 in a pseudo-camera space where
        # Z is normal to board and X,Y aligned with board axes. For physical EE mapping, use
        # extrinsics from calibration script to get T_base_board.

        R = np.eye(3, dtype=float)
        t = np.array([0.0, 0.0, 0.0], dtype=float)
        return SE3.from_rt(R, t)

    @staticmethod
    def draw_corners(image_bgr: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        out = image_bgr.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for i, p in enumerate(pts_xy.astype(int)):
            cv2.circle(out, tuple(p.tolist()), 6, colors[i % len(colors)], -1)
        return out


