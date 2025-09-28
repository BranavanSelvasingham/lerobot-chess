#!/usr/bin/env python

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import argparse
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.processor.pipelines.chess_pick_place import build_chess_pick_place_pipeline
from lerobot.perception.chess.board_model import BoardModel
from lerobot.configs.chessboard import ChessBoardParams
from lerobot.utils.utils import log_say


@dataclass
class MoveConfig:
    port: str
    camera_name: str = "front"
    urdf_path: str = "./SO101/so101_new_calib.urdf"
    from_square: str = "e2"
    to_square: str = "e4"
    hover_height_m: float = 0.08
    transit_height_m: float = 0.15
    skip_calibration: bool = False


def _parse_args() -> MoveConfig:
    p = argparse.ArgumentParser(description="Execute a single chess move with SO-101")
    p.add_argument("--port", required=True, help="Serial port for SO-101, e.g. /dev/tty.usbmodemXXXX")
    p.add_argument("--urdf_path", default="./SO101/so101_new_calib.urdf")
    p.add_argument("--from", dest="from_square", default="e2")
    p.add_argument("--to", dest="to_square", default="e4")
    p.add_argument("--hover_height_m", type=float, default=0.08)
    p.add_argument("--transit_height_m", type=float, default=0.15)
    p.add_argument("--skip-calibration", action="store_true", help="Skip robot calibration")
    a = p.parse_args()
    return MoveConfig(
        port=a.port,
        urdf_path=a.urdf_path,
        from_square=a.from_square,
        to_square=a.to_square,
        hover_height_m=a.hover_height_m,
        transit_height_m=a.transit_height_m,
        skip_calibration=getattr(a, 'skip_calibration', False),
    )


def main(cfg: MoveConfig | None = None) -> None:
    if cfg is None:
        cfg = _parse_args()
    robot_cfg = SO101FollowerConfig(port=cfg.port, id="so101_chess", cameras={}, use_degrees=True)
    robot = SO101Follower(robot_cfg)
    robot.connect(calibrate=not cfg.skip_calibration, skip_firmware_check=True)

    # Load kinematics
    kin = RobotKinematics(
        urdf_path=cfg.urdf_path, target_frame_name="gripper_frame_link", joint_names=list(robot.bus.motors.keys())
    )

    # Load board model with T_base_board already set via calibration script
    board_model = BoardModel(params=ChessBoardParams())
    # For now require saved transform at robot.calibration_dir / "chess_board_model.json"
    bm_path = robot.calibration_dir / "chess_board_model.json"
    if bm_path.exists():
        board_model = BoardModel.load(bm_path)
    else:
        raise SystemExit(f"Board model not found at {bm_path}. Run calibration first.")

    pipeline = build_chess_pick_place_pipeline(
        board_model=board_model,
        kinematics=kin,
        motor_names=list(robot.bus.motors.keys()),
        hover_height_m=cfg.hover_height_m,
        transit_height_m=cfg.transit_height_m,
    )

    # Seed complementary data with move request
    comp = {"chess_move": {"from": cfg.from_square, "to": cfg.to_square}}
    transition = {"observation": robot.get_observation(), "action": {}, "complementary_data": comp}
    for step in pipeline.steps:
        transition = step(transition)

    waypoints = transition["complementary_data"]["ee_waypoints_sequence"]
    dwell = float(transition["complementary_data"]["ee_waypoint_dwell_s"])

    exec_segment = pipeline.exec_segment

    for wp in waypoints:
        # Reset transition with latest observation and this waypoint as action
        transition = {"observation": robot.get_observation(), "action": dict(wp)}
        for step in exec_segment.steps:
            transition = step(transition)
        action = transition["action"]
        robot.send_action(action)
        time.sleep(dwell)

    log_say("Move completed. Returning to idle.")


if __name__ == "__main__":
    main()


