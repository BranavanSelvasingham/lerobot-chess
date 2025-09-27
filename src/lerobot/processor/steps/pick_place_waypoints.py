#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import ComplementaryDataProcessorStep


@dataclass
class PickPlaceWaypointPlanner(ComplementaryDataProcessorStep):
    vertical_only: bool = True
    transit_height_m: float = 0.15
    grasp_close: float = 80.0
    open_value: float = 5.0
    dwell_s: float = 0.2

    def complementary_data(self, complementary_data: dict) -> dict:
        req = complementary_data
        keys = [
            "ee_waypoint_src_hover_xyz",
            "ee_waypoint_src_touch_xyz",
            "ee_waypoint_dst_hover_xyz",
            "ee_waypoint_dst_touch_xyz",
        ]
        if not all(k in req for k in keys):
            return complementary_data

        src_h = np.array(req["ee_waypoint_src_hover_xyz"], dtype=float)
        src_t = np.array(req["ee_waypoint_src_touch_xyz"], dtype=float)
        dst_h = np.array(req["ee_waypoint_dst_hover_xyz"], dtype=float)
        dst_t = np.array(req["ee_waypoint_dst_touch_xyz"], dtype=float)

        # high transit waypoint
        high = np.array([src_h[0], src_h[1], max(src_h[2], dst_h[2], self.transit_height_m)], dtype=float)

        waypoints: list[dict[str, float]] = []

        def ee_pose(xyz: np.ndarray, grip: float) -> dict[str, float]:
            return {
                "ee.x": float(xyz[0]),
                "ee.y": float(xyz[1]),
                "ee.z": float(xyz[2]),
                "ee.wx": 0.0,
                "ee.wy": 0.0,
                "ee.wz": 0.0,
                "ee.gripper_pos": float(grip),
            }

        # Sequence: hover src (open) -> touch src (open) -> close -> hover src -> high -> hover dst -> touch dst -> open -> hover dst
        waypoints.append(ee_pose(src_h, self.open_value))
        waypoints.append(ee_pose(src_t, self.open_value))
        waypoints.append(ee_pose(src_t, self.grasp_close))
        waypoints.append(ee_pose(src_h, self.grasp_close))
        waypoints.append(ee_pose(high, self.grasp_close))
        waypoints.append(ee_pose(dst_h, self.grasp_close))
        waypoints.append(ee_pose(dst_t, self.grasp_close))
        waypoints.append(ee_pose(dst_t, self.open_value))
        waypoints.append(ee_pose(dst_h, self.open_value))

        out = dict(complementary_data)
        out["ee_waypoints_sequence"] = waypoints
        out["ee_waypoint_dwell_s"] = float(self.dwell_s)
        return out


