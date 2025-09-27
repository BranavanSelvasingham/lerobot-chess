#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class OccupancyDetector:
    """Minimal background-subtraction-based occupancy over 8x8 grid.

    Provide a reference image of an empty board (aligned), then for a new frame, compute a boolean
    (8,8) mask where True indicates a piece present.
    """

    square_size_px: int
    grid_offset_xy: tuple[int, int]
    threshold: float = 25.0

    empty_reference_gray: Optional[np.ndarray] = None  # HxW uint8

    def set_empty_reference(self, image_bgr: np.ndarray) -> None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self.empty_reference_gray = gray

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.empty_reference_gray is None:
            raise ValueError("empty_reference_gray not set. Call set_empty_reference first.")
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, self.empty_reference_gray)
        # Simple blur + threshold for robustness
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, binm = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        occ = np.zeros((8, 8), dtype=bool)
        ox, oy = self.grid_offset_xy
        s = self.square_size_px
        for r in range(8):
            for c in range(8):
                x0 = ox + c * s
                y0 = oy + r * s
                roi = binm[y0 : y0 + s, x0 : x0 + s]
                # occupancy if sufficient foreground pixels
                occ[r, c] = (np.count_nonzero(roi) / (s * s)) > 0.05
        return occ


