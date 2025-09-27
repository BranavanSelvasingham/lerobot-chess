#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import RobotActionProcessorStep


@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    kinematics: RobotKinematics
    motor_names: list[str]
    initial_guess_current_joints: bool = True
    position_weight: float = 1.0
    orientation_weight: float = 0.01

    _q_curr: np.ndarray | None = None

    def action(self, action: dict[str, float]) -> dict[str, float]:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos", None)

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is required for IK initial guess")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if self.initial_guess_current_joints or self._q_curr is None:
            self._q_curr = q_raw

        T = np.eye(4, dtype=float)
        from lerobot.utils.geometry import rodrigues

        T[:3, :3] = rodrigues([wx, wy, wz])
        T[:3, 3] = [x, y, z]

        q_target = self.kinematics.inverse_kinematics(
            self._q_curr, T, position_weight=self.position_weight, orientation_weight=self.orientation_weight
        )
        self._q_curr = q_target

        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
        if gripper_pos is not None:
            action["gripper.pos"] = float(gripper_pos)
        return action


