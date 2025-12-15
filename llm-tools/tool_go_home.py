from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_toolkit import KinematicsTools


_SO101_AND_GRIPPER: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "go_home",
        "description": "Move the robot to the saved home pose.",
        "strict": False,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    }


def execute(tools: "KinematicsTools", args: dict[str, Any]) -> dict[str, Any]:
    _ = args
    with tools._lock:
        if not tools.home_position_path.is_file():
            raise RuntimeError(f"Home position not found at {tools.home_position_path}")

        obj = json.loads(tools.home_position_path.read_text())
        pos = obj.get("motor_positions") or {}

        targets: dict[str, float] = {}
        for j in _SO101_AND_GRIPPER:
            if j in pos:
                targets[j] = float(pos[j])

        if not targets:
            raise RuntimeError("Home position file has no motor_positions")

        tools._send_joint_targets_deg(targets)
        # Reset delta-move accumulator so future delta commands start from the new physical pose.
        try:
            tools._ee_cmd_xyz_m = None  # type: ignore[attr-defined]
        except Exception:
            pass
        return {"ok": True, "targets_deg": targets}
