#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.processor.steps.chess_square_to_ee_pose import ChessSquareToEEPose
from lerobot.processor.steps.ee_bounds_and_safety import EEBoundsAndSafety
from lerobot.processor.steps.gripper_setpoint_to_joint import GripperSetpointToJoint
from lerobot.processor.steps.inverse_kinematics import InverseKinematicsEEToJoints
from lerobot.processor.steps.pick_place_waypoints import PickPlaceWaypointPlanner


def build_chess_pick_place_pipeline(
    board_model,
    kinematics: RobotKinematics,
    motor_names: list[str],
    hover_height_m: float = 0.08,
    transit_height_m: float = 0.15,
    ee_bounds: dict | None = None,
) -> RobotProcessorPipeline[dict, dict]:
    ee_bounds = ee_bounds or {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}

    steps = [
        ChessSquareToEEPose(board_model=board_model, hover_height_m=hover_height_m),
        PickPlaceWaypointPlanner(transit_height_m=transit_height_m),
    ]

    # The execution of waypoints is handled externally by iterating produced sequence
    # and passing each waypoint through the safety -> IK -> gripper chain.

    safety = EEBoundsAndSafety(
        min_xyz=tuple(ee_bounds["min"]), max_xyz=tuple(ee_bounds["max"]), max_ee_step_m=0.05
    )
    ik = InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=motor_names)
    grip = GripperSetpointToJoint()

    # Build a pipeline segment for single-waypoint execution
    exec_segment = RobotProcessorPipeline(steps=[safety, grip, ik])

    # Return both planning steps and execution segment so caller can loop
    pipeline = RobotProcessorPipeline(steps=steps)
    pipeline.exec_segment = exec_segment  # attach for convenience
    return pipeline


