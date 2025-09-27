#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.processor.core import RobotAction
from lerobot.processor.pipeline import RobotActionProcessorStep


@dataclass
class EEBoundsAndSafety(RobotActionProcessorStep):
    """Clamp EE pose deltas and absolute bounds before IK conversion.

    Expected input keys in action:
      ee.x, ee.y, ee.z  (meters)
      ee.wx, ee.wy, ee.wz (rad, Rodrigues vector components)
      ee.gripper_pos (0..100)
    """

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    max_ee_step_m: float = 0.05
    max_rot_step_rad: float = 0.3

    _prev_xyz: np.ndarray | None = None
    _prev_rot: np.ndarray | None = None

    def action(self, action: RobotAction) -> RobotAction:
        x = float(action.get("ee.x", 0.0))
        y = float(action.get("ee.y", 0.0))
        z = float(action.get("ee.z", 0.0))
        wx = float(action.get("ee.wx", 0.0))
        wy = float(action.get("ee.wy", 0.0))
        wz = float(action.get("ee.wz", 0.0))

        xyz = np.array([x, y, z], dtype=float)
        rot = np.array([wx, wy, wz], dtype=float)

        # Absolute clamp
        xyz = np.clip(xyz, np.array(self.min_xyz), np.array(self.max_xyz))

        # Step clamp
        if self._prev_xyz is not None:
            d = xyz - self._prev_xyz
            n = np.linalg.norm(d)
            if n > self.max_ee_step_m and n > 1e-9:
                xyz = self._prev_xyz + d * (self.max_ee_step_m / n)

        if self._prev_rot is not None:
            d = rot - self._prev_rot
            n = np.linalg.norm(d)
            if n > self.max_rot_step_rad and n > 1e-9:
                rot = self._prev_rot + d * (self.max_rot_step_rad / n)

        self._prev_xyz = xyz
        self._prev_rot = rot

        action.update({
            "ee.x": float(xyz[0]),
            "ee.y": float(xyz[1]),
            "ee.z": float(xyz[2]),
            "ee.wx": float(rot[0]),
            "ee.wy": float(rot[1]),
            "ee.wz": float(rot[2]),
        })
        return action


