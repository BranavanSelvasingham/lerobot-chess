#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

import draccus
import numpy as np

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.perception.chess.board_model import BoardModel
from lerobot.perception.chess.board_pose_estimator import BoardPoseEstimator
from lerobot.configs.chessboard import ChessBoardParams
from lerobot.utils.utils import log_say


@dataclass
class CalibrateBoardConfig:
    port: str
    urdf_path: str = "./SO101/so101_new_calib.urdf"  # reserved for future use
    square_size_mm: float = 50.0


def main(cfg: CalibrateBoardConfig = draccus.run(CalibrateBoardConfig)) -> None:
    robot_cfg = SO101FollowerConfig(port=cfg.port, id="so101_chess", cameras={}, use_degrees=True)
    robot = SO101Follower(robot_cfg)
    robot.connect(calibrate=True)

    # Capture one frame from any connected camera; here we expect an arm-mounted camera to exist in config
    if not robot.cameras:
        raise SystemExit("No cameras configured on robot. Add arm camera to SO101FollowerConfig.cameras.")
    cam = next(iter(robot.cameras.values()))
    frame = cam.read()

    # Prompt user to click 4 board corners elsewhere (UI not implemented). Placeholder uses center quad.
    h, w = frame.shape[:2]
    s = min(h, w) * 0.6
    cx, cy = w / 2.0, h / 2.0
    image_corners = np.array(
        [
            [cx - s / 2, cy + s / 2],  # a1
            [cx + s / 2, cy + s / 2],  # h1
            [cx + s / 2, cy - s / 2],  # h8
            [cx - s / 2, cy - s / 2],  # a8
        ],
        dtype=float,
    )

    estimator = BoardPoseEstimator(square_size_mm=cfg.square_size_mm)
    T_cam_board = estimator.estimate_from_corners(image_corners)

    # For now we cannot compute T_base_board without a known T_base_camera; leave identity and let user edit later
    board_model = BoardModel(params=ChessBoardParams(square_size_mm=cfg.square_size_mm), T_base_board=T_cam_board)

    out_path = robot.calibration_dir / "chess_board_model.json"
    board_model.save(out_path)
    log_say(f"Saved provisional board model to {out_path}. Replace T_base_board with calibrated transform later.")


if __name__ == "__main__":
    main()


