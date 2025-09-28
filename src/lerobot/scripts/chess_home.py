#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.utils import log_say


@dataclass
class HomeConfig:
    port: str
    save: bool = False
    go: bool = True


def _parse_args() -> HomeConfig:
    p = argparse.ArgumentParser(description="Save or go to a home joint pose")
    p.add_argument("--port", required=True)
    p.add_argument("--save", action="store_true")
    p.add_argument("--go", action="store_true")
    a = p.parse_args()
    # Default to go if neither flag set
    if not a.save and not a.go:
        a.go = True
    return HomeConfig(port=a.port, save=a.save, go=a.go)


def main(cfg: HomeConfig | None = None) -> None:
    if cfg is None:
        cfg = _parse_args()
    robot_cfg = SO101FollowerConfig(port=cfg.port, id="so101_chess", cameras={}, use_degrees=True)
    robot = SO101Follower(robot_cfg)
    robot.connect(calibrate=True)

    home_path = robot.calibration_dir / "home_joints.npy"

    if cfg.save:
        obs = robot.get_observation()
        q = np.array([obs[f"{m}.pos"] for m in robot.bus.motors.keys()], dtype=float)
        np.save(home_path, q)
        log_say(f"Saved home to {home_path}")
        return

    if cfg.go:
        if not home_path.exists():
            raise SystemExit("No saved home pose. Run with --save first.")
        q = np.load(home_path)
        action = {f"{m}.pos": float(q[i]) for i, m in enumerate(robot.bus.motors.keys())}
        robot.send_action(action)
        log_say("Moved to home pose.")


if __name__ == "__main__":
    main()


