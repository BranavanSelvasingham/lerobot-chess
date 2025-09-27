#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

from lerobot.processor.pipeline import RobotActionProcessorStep


@dataclass
class GripperSetpointToJoint(RobotActionProcessorStep):
    """Pass-through mapping of ee.gripper_pos to gripper.pos joint space (0..100)."""

    def action(self, action: dict[str, float]) -> dict[str, float]:
        gp = action.get("ee.gripper_pos")
        if gp is not None:
            action["gripper.pos"] = float(gp)
        return action


