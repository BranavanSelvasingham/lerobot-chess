from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_toolkit import KinematicsTools


_SO101_JOINTS: list[str] = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "read_joints",
        "description": (
            "Read the robot's current joint positions. "
            "Body joints are in degrees. Gripper is in 0..100 units (robot-specific semantics)."
        ),
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "include_gripper": {"type": "boolean", "description": "Include gripper readback (default true)."},
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def execute(tools: "KinematicsTools", args: dict[str, Any]) -> dict[str, Any]:
    with tools._lock:
        tools._require_robot()
        include_gripper = bool(args.get("include_gripper", True))

        names = list(_SO101_JOINTS)
        if include_gripper:
            names.append("gripper")

        try:
            q = tools._read_joints_deg(names)  # type: ignore[attr-defined]
            return {
                "ok": True,
                "torque_disabled": bool(getattr(tools, "torque_disabled", False)),
                "joints": q,
            }
        except Exception as e:
            return {
                "ok": False,
                "torque_disabled": bool(getattr(tools, "torque_disabled", False)),
                "error": str(e),
            }
