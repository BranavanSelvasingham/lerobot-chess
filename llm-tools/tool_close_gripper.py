"""Tool: close_gripper - close the gripper to grasp a piece."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_toolkit import KinematicsTools


def schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "close_gripper",
        "description": "Close the gripper to grasp a chess piece.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }


def execute(tools: "KinematicsTools", args: dict[str, Any]) -> dict[str, Any]:
    _ = args
    with tools._lock:
        # Close enough to grip a piece but not fully closed
        tools._send_joint_targets_deg({"gripper": 20.0})
        return {"ok": True, "gripper_percent": 20.0, "action": "closed"}
