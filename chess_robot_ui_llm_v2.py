#!/usr/bin/env python3

"""Minimal Chess Robot UI (LLM v2)

This is a ground-up rewrite of `chess_robot_ui_llm.py` focused on two things only:
- Live camera view
- Tool-call view/log (manual + optional LLM tool-calling)

Kinematics-based tools are implemented via `lerobot.model.kinematics.RobotKinematics`
(placho/IK) and execute joint-space commands on the SO-101 follower arm.

Run:
  python chess_robot_ui_llm_v2.py --port /dev/tty.usbmodemXXXX

Notes:
- Requires a URDF available on disk for IK.
- Uses calibration assets in HF LeRobot cache by default.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Make `src/` importable when running from repo root.
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_LLM_TOOLS_DIR = _REPO_ROOT / "llm-tools"
if _LLM_TOOLS_DIR.exists() and str(_LLM_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_TOOLS_DIR))

from PySide6.QtCore import QFileSystemWatcher, QThread, QTimer, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from openai import OpenAI  # type: ignore

    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

import cv2

from lerobot.cameras.opencv.configuration_opencv import ColorMode, OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from llm_toolkit import AppConfig, KinematicsTools  # pyright: ignore[reportMissingImports]


class ToolExecutionThread(QThread):
    log_line = Signal(str)
    finished_ok = Signal(bool)

    def __init__(self, tools: KinematicsTools, tool_name: str, tool_args: dict[str, Any]):
        super().__init__()
        self.tools = tools
        self.tool_name = tool_name
        self.tool_args = dict(tool_args or {})

    def run(self) -> None:
        try:
            self.log_line.emit(f"TOOL {self.tool_name} args={json.dumps(self.tool_args)}")
            result = self.tools.execute_tool(self.tool_name, self.tool_args)
            self.log_line.emit(json.dumps(result, indent=2))
            self.finished_ok.emit(bool(result.get("ok", False)))
        except Exception as e:
            self.log_line.emit(json.dumps({"ok": False, "error": str(e)}, indent=2))
            self.finished_ok.emit(False)


class LLMThread(QThread):
    log_line = Signal(str)
    finished_ok = Signal(bool)

    def __init__(
        self,
        *,
        ui: "ChessRobotUILLMV2",
        llm_client: Any,
        model: str,
        tools: KinematicsTools,
        command: str,
        max_steps: int = 20,
    ):
        super().__init__()
        self.ui = ui
        self.llm_client = llm_client
        self.model = model
        self.tools_impl = tools
        self.command = command
        self.max_steps = int(max_steps)

    def run(self) -> None:
        try:
            tool_schemas = self.tools_impl.tool_schemas()
            prev_response_id: str | None = None

            # Log accumulator for conversation context
            log_history: list[str] = []

            # Capture initial camera image
            initial_image = self.ui.capture_frame_base64()
            log_history.append(f"USER: {self.command}")

            # Build initial user message with optional image
            user_content: list[dict[str, Any]] = [
                {"type": "input_text", "text": self.command}
            ]

            # Always include current motor positions so the model can do reliable joint-space control.
            try:
                joints_now = self.tools_impl.execute_tool("read_joints", {"include_gripper": True})
                joints_payload = joints_now
                if isinstance(joints_now, dict) and isinstance(joints_now.get("joints"), dict):
                    joints_payload = joints_now["joints"]
                user_content.append(
                    {
                        "type": "input_text",
                        "text": "CURRENT_JOINTS (degrees; gripper 0..100):\n"
                        + json.dumps(joints_payload, indent=2),
                    }
                )
                self.log_line.emit("[LLM input includes current joint positions]")
                self.log_line.emit("[LLM joint snapshot]\n" + json.dumps(joints_payload, indent=2))
            except Exception:
                self.log_line.emit("[LLM input joint snapshot failed]")

            # Include a compact URDF summary (dimensions/axes/limits) ONCE at the start.
            try:
                urdf_summary = self.tools_impl.urdf_summary()
                if urdf_summary and isinstance(urdf_summary, dict):
                    user_content.append(
                        {
                            "type": "input_text",
                            "text": "URDF_SUMMARY (axes/limits/link offsets; model-derived):\n"
                            + json.dumps(urdf_summary, indent=2),
                        }
                    )
                    self.log_line.emit("[LLM input includes URDF summary]")
            except Exception:
                self.log_line.emit("[LLM input URDF summary unavailable]")

            # Include URDF-derived local joint effects (FK finite-difference).
            try:
                effects = self.tools_impl.joint_effects_mm_per_deg()
                user_content.append(
                    {
                        "type": "input_text",
                        "text": "JOINT_EFFECTS_MM_PER_DEG (URDF-derived FK sensitivity at CURRENT_JOINTS):\n"
                        + json.dumps(effects, indent=2),
                    }
                )
                self.log_line.emit("[LLM input includes joint effects]")
            except Exception:
                self.log_line.emit("[LLM input joint effects unavailable]")

            if initial_image:
                user_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{initial_image}",
                })
                # Avoid printing base64; just show size for observability.
                self.log_line.emit(f"[LLM input includes initial image (base64 chars={len(initial_image)})]")
            else:
                self.log_line.emit("[LLM input has no image available yet]")

            input_items: Any = [{"role": "user", "content": user_content}]

            # System instructions mentioning vision capability
            instructions = (
                "You control a real SO-101 robot arm for chess piece manipulation. "
                "You receive camera images showing the current board state after each action.\n\n"

                "AUTONOMOUS EXECUTION POLICY:\n"
                "- You have access to all tools the user has access to.\n"
                "- Do NOT ask the user questions or request extra input; infer reasonable defaults and proceed.\n"
                "- Use the camera images and tool results to verify progress and adjust.\n"
                "- Execute multi-step sequences as needed until the goal is achieved.\n"
                "- If the goal seems unreachable (e.g., 3 consecutive motion failures / large persistent error / overload), call go_home and retry.\n"
                "- Only stop early if a safety issue occurs (overload / torque-off / unreachable), then report what happened.\n\n"

                "CAMERA SETUP:\n"
                "- The camera is mounted ON THE GRIPPER (eye-in-hand configuration)\n"
                "- The gripper tongs/tips are visible at the bottom of the camera image\n"
                "- When you see a piece, it is viewed from above through the gripper\n"
                "- To grab a piece: position it BETWEEN the two gripper tongs in the image\n"
                "- The piece should appear centered between the tongs before closing\n\n"
                
                "CHESS BOARD:\n"
                "- Each square is approximately 1 inch × 1 inch (~25mm × 25mm)\n"
                "- Squares are labeled a1-h8 (files a-h, ranks 1-8)\n"
                "- The board calibration may have slight offsets\n\n"
                
                "AVAILABLE TOOLS:\n"
                "1. go_home: Returns robot to saved home position. Use to reset or start fresh.\n"
                "2. move_gripper_delta: Move end-effector by delta amounts:\n"
                "   - dx_mm: radial delta (+ = away from base, - = toward base)\n"
                "   - dy_mm: tangential delta (left/right arc)\n"
                "   - dz_mm: vertical delta (+ = up, - = down)\n"
                "   - Typical moves: 10-50mm. Max ±200mm.\n"
                "3. move_piece: Pick piece from one square, place on another:\n"
                "   - from_square, to_square: e.g., 'e2', 'e4'\n"
                "   - Executes full pick-and-place sequence with hover waypoints\n"
                "4. set_gripper_percent: Control gripper opening:\n"
                "   - IMPORTANT: On this robot, 0 = fully CLOSED, 100 = fully OPEN\n"
                "   - Use ~90-100 for open, ~10-30 for gripping pieces\n\n"
                "5. look_around: Move the gripper-mounted camera one step to search (left/right/up/down).\n"
                "   - Call repeatedly one step at a time and stop once the target is in view.\n\n"
                "6. move_joints: Direct joint-space control (degrees for body joints, 0..100 for gripper).\n"
                "7. read_joints: Read current joints (degrees + gripper 0..100).\n"
                "8. set_all_joints: Set ALL motor targets at once (absolute pose in joint space).\n"
                "9. read_motor_diagnostics: Read motor load/current/temp/voltage/status.\n"
                "10. disable_torque / enable_torque: Emergency stop and recovery.\n\n"
                
                "STATE FEEDBACK:\n"
                "- Every message includes CURRENT_JOINTS with all motor positions.\n"
                "- Use CURRENT_JOINTS + images to close the loop when moves are unreliable.\n\n"

                "JOINT CHEAT SHEET (SO-101, APPROXIMATE):\n"
                "- shoulder_pan: rotates the whole arm left/right (moves camera view sideways).\n"
                "- shoulder_lift: raises/lowers the upper arm (affects height + reach).\n"
                "- elbow_flex: bends/extends the elbow (affects reach to/from the board).\n"
                "- wrist_flex: pitches the wrist (changes camera tilt and gripper approach angle).\n"
                "- wrist_roll: rolls the gripper about its axis (rotates the tongs in the image).\n"
                "- gripper: 0=closed, 100=open.\n"
                "NOTE: The SIGN of each joint may be opposite of your intuition. If unsure, probe with ±2–5° and observe.\n\n"

                "HOW TO USE set_all_joints EFFECTIVELY:\n"
                "- Start from CURRENT_JOINTS, copy it, then change ONLY 1–2 joints by a small amount.\n"
                "- Call set_all_joints with all 6 targets. The tool clamps per-call deltas for safety.\n"
                "- After each call, look at the new image + CURRENT_JOINTS to see what changed.\n"
                "- Build a mental mapping: which joint moves the piece toward the tong center.\n"
                "- If you get lost or near limits, call go_home and retry.\n\n"
                
                "GRABBING A PIECE:\n"
                "- First, ensure gripper is open (set_gripper_percent ~90-100)\n"
                "- Use move_gripper_delta to position the piece between the tongs\n"
                "- The piece should be centered horizontally between the two tong tips\n"
                "- Lower the gripper (dz_mm negative) so tongs are at piece height\n"
                "- Close gripper (set_gripper_percent ~10-30) to grasp\n"
                "- Verify grip in the camera image before lifting\n"
                "- Do NOT claim you gripped/lifted a piece unless you visually confirm it in the image\n\n"
                
                "ACCURACY LIMITATIONS:\n"
                "- Position accuracy is ~5-10mm due to motor backlash and IK solving\n"
                "- Small moves (<10mm) may not register due to motor deadband\n"
                "- Repeated small deltas can accumulate drift\n"
                "- After tool execution, check the updated image to verify the action\n"
                "- For motion tools: if the result has ok=false / ik_ok=false or a large final_error_mm, treat it as a missed move and retry (or switch to move_joints)\n"
                "- If position seems off, consider going home and retrying\n\n"
                
                "BEST PRACTICES:\n"
                "- Prefer medium-sized moves (20-50mm) over many tiny ones\n"
                "- Use the camera image to visually align the piece between tongs\n"
                "- Use move_gripper_delta for fine positioning adjustments\n"
                "- If Cartesian moves are unreliable, switch to joint-space nudges using move_joints + camera feedback\n"
                "- Always verify results in the camera image before proceeding\n"
                "- If unsure about where the pawn is, use look_around to scan; do not ask the user\n"
            )

            for step in range(self.max_steps):
                # Log what we're about to send (high signal, no giant blobs)
                in_has_image = False
                try:
                    # Detect image parts in the outbound items (best-effort)
                    for it in input_items:
                        if isinstance(it, dict) and it.get("role") == "user":
                            content = it.get("content", [])
                            if isinstance(content, list):
                                in_has_image = any(
                                    isinstance(c, dict) and c.get("type") in ("input_image", "image_url") for c in content
                                )
                except Exception:
                    pass
                self.log_line.emit(
                    f"[LLM request step={step+1}/{self.max_steps} model={self.model} image={in_has_image} prev_response_id={'set' if prev_response_id else 'none'}]"
                )

                resp = self.llm_client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=input_items,
                    tools=tool_schemas,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    max_tool_calls=1,
                    max_output_tokens=450,
                    previous_response_id=prev_response_id,
                )

                prev_response_id = str(getattr(resp, "id", prev_response_id) or "") or prev_response_id

                txt = str(getattr(resp, "output_text", "") or "").strip()
                if txt:
                    self.log_line.emit(f"LLM: {txt}")
                    log_history.append(f"LLM: {txt}")
                else:
                    self.log_line.emit("[LLM: (no text output)]")

                calls = [it for it in getattr(resp, "output", []) if getattr(it, "type", None) == "function_call"]
                if not calls:
                    self.finished_ok.emit(True)
                    return

                call = calls[0]
                name = str(getattr(call, "name", ""))
                raw_args = getattr(call, "arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    args = {}
                if not isinstance(args, dict):
                    args = {}

                self.log_line.emit(f"LLM->TOOL {name} args={json.dumps(args)}")
                log_history.append(f"TOOL {name}: {json.dumps(args)}")

                result = self.tools_impl.execute_tool(name, args)
                result_str = json.dumps(result, indent=2)
                self.log_line.emit(result_str)
                log_history.append(f"RESULT: {result_str}")

                # Capture post-action image
                post_image = self.ui.capture_frame_base64()

                # Build context from recent log entries (last 10)
                context_text = "\n".join(log_history[-10:])
                # Show exactly what context text we're about to send back (no images / no base64).
                self.log_line.emit("[LLM context sent after tool]\n" + context_text)

                # Feed tool output back to the model with image
                input_items = [
                    {
                        "type": "function_call_output",
                        "call_id": str(getattr(call, "call_id", "")),
                        "output": json.dumps(result),
                    }
                ]

                # Always include current motor positions in the next prompt (even if image is missing).
                joints_text = ""
                try:
                    joints_now = self.tools_impl.execute_tool("read_joints", {"include_gripper": True})
                    joints_payload = joints_now
                    if isinstance(joints_now, dict) and isinstance(joints_now.get("joints"), dict):
                        joints_payload = joints_now["joints"]
                    joints_text = "CURRENT_JOINTS (degrees; gripper 0..100):\n" + json.dumps(
                        joints_payload, indent=2
                    )
                    self.log_line.emit("[LLM input includes current joint positions]")
                    self.log_line.emit("[LLM joint snapshot]\n" + json.dumps(joints_payload, indent=2))
                except Exception:
                    joints_text = "CURRENT_JOINTS: [error reading joints]"

                effects_text = ""
                try:
                    effects = self.tools_impl.joint_effects_mm_per_deg()
                    effects_text = "JOINT_EFFECTS_MM_PER_DEG:\n" + json.dumps(effects, indent=2)
                except Exception:
                    effects_text = "JOINT_EFFECTS_MM_PER_DEG: [unavailable]"

                user_parts: list[dict[str, Any]] = [
                    {
                        "type": "input_text",
                        "text": f"[Action complete. Recent log:\n{context_text}]\n\n{joints_text}\n\n{effects_text}",
                    }
                ]
                if post_image:
                    user_parts.append(
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{post_image}"}
                    )
                    self.log_line.emit(f"[LLM input includes post-action image (base64 chars={len(post_image)})]")
                else:
                    self.log_line.emit("[LLM input has no post-action image available]")

                input_items.append({"role": "user", "content": user_parts})

            self.log_line.emit(json.dumps({"ok": False, "error": f"LLM exceeded max_steps={self.max_steps}"}, indent=2))
            self.finished_ok.emit(False)
        except Exception as e:
            self.log_line.emit(json.dumps({"ok": False, "error": str(e)}, indent=2))
            self.finished_ok.emit(False)


class ChessRobotUILLMV2(QMainWindow):
    append_log_signal = Signal(str)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        # AppConfig is frozen; keep UI-selected values separately.
        self._selected_model: str = str(getattr(cfg, "model", "gpt-4o-mini"))

        self.setWindowTitle("Chess Robot UI (LLM v2) - Camera + Tools")
        self.resize(1280, 720)

        self.tools = KinematicsTools(cfg)
        self._torque_disabled_ui: bool = False

        self._camera: OpenCVCamera | None = None
        self._last_frame_bgr: np.ndarray | None = None

        self._llm_init_error: str | None = None
        self._llm_client: Any | None = None

        # Load API key from repo-local .env if present (optional).
        # Many users keep OPENAI_API_KEY in a .env file but don't export it into the shell.
        try:
            if os.getenv("OPENAI_API_KEY") is None:
                env_path = _REPO_ROOT / ".env"
                if env_path.is_file():
                    for raw in env_path.read_text().splitlines():
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and (os.getenv(k) is None):
                            os.environ[k] = v
        except Exception:
            # If parsing fails, just fall back to environment variables.
            pass

        if _OPENAI_AVAILABLE:
            api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self._llm_client = OpenAI(api_key=api_key)
                except Exception as e:
                    self._llm_init_error = str(e)
                    self._llm_client = None

        self._tool_thread: ToolExecutionThread | None = None
        self._llm_thread: LLMThread | None = None

        self.append_log_signal.connect(self._append_log)

        self._build_ui()
        self._setup_camera()
        self._setup_file_watcher()

        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._tick_camera)
        self._camera_timer.start(int(1000 / max(5, int(cfg.camera_fps))))

        self._refresh_status()

    # -----------------------------
    # UI
    # -----------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Restart banner (hidden by default)
        self.restart_banner = QFrame()
        self.restart_banner.setStyleSheet(
            "QFrame { background-color: #1a3a5c; border-bottom: 1px solid #2d5a8a; }"
        )
        self.restart_banner.setVisible(False)
        banner_layout = QHBoxLayout(self.restart_banner)
        banner_layout.setContentsMargins(10, 6, 10, 6)
        banner_label = QLabel("🔄 Code changes detected.")
        banner_label.setStyleSheet("color: #58a6ff; font-weight: bold;")
        banner_layout.addWidget(banner_label)
        banner_layout.addStretch()
        self.restart_btn = QPushButton("Restart")
        self.restart_btn.setStyleSheet(
            "QPushButton { background-color: #238636; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #2ea043; }"
        )
        self.restart_btn.clicked.connect(self._restart_app)
        banner_layout.addWidget(self.restart_btn)
        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setStyleSheet(
            "QPushButton { background-color: #30363d; color: #c9d1d9; padding: 4px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #484f58; }"
        )
        dismiss_btn.clicked.connect(lambda: self.restart_banner.setVisible(False))
        banner_layout.addWidget(dismiss_btn)
        root_layout.addWidget(self.restart_banner)

        # Main content area
        content = QWidget()
        root_layout.addWidget(content, 1)
        layout = QHBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Camera panel
        cam_group = QGroupBox("Camera")
        cam_layout = QVBoxLayout(cam_group)
        cam_layout.setContentsMargins(10, 10, 10, 10)

        self.camera_label = QLabel("(camera not connected)")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background: #0b0f14; color: #c9d1d9; border: 1px solid #30363d;")
        cam_layout.addWidget(self.camera_label, 1)

        self.camera_status = QLabel("Camera: --")
        self.camera_status.setStyleSheet("color: #8b949e;")
        cam_layout.addWidget(self.camera_status)

        splitter.addWidget(cam_group)

        # Tools panel
        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout(tools_group)
        tools_layout.setContentsMargins(10, 10, 10, 10)
        tools_layout.setSpacing(10)

        self.status_label = QLabel("Robot: -- | Kinematics: -- | Board: --")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #c9d1d9;")
        tools_layout.addWidget(self.status_label)

        # Torque + diagnostics row
        torque_row = QHBoxLayout()
        self.torque_off_btn = QPushButton("STOP (Torque OFF)")
        self.torque_off_btn.setStyleSheet(
            "QPushButton { background-color: #da3633; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; }"
            "QPushButton:hover { background-color: #f85149; }"
        )
        self.torque_off_btn.clicked.connect(self._torque_off)
        torque_row.addWidget(self.torque_off_btn)

        self.torque_on_btn = QPushButton("Torque ON")
        self.torque_on_btn.setStyleSheet(
            "QPushButton { background-color: #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #484f58; }"
        )
        self.torque_on_btn.clicked.connect(self._torque_on)
        torque_row.addWidget(self.torque_on_btn)

        self.diag_btn = QPushButton("Read Motor Diagnostics")
        self.diag_btn.clicked.connect(self._read_motor_diagnostics)
        torque_row.addWidget(self.diag_btn)
        torque_row.addStretch()
        tools_layout.addLayout(torque_row)

        # LLM controls
        llm_row = QHBoxLayout()

        # Model selector
        model_label = QLabel("Model:")
        llm_row.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            # GPT-5.2 series (latest)
            "gpt-5.2",
            "gpt-5.2-pro",
            "gpt-5.2-chat-latest",
            # GPT-5 family
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            # GPT-4o series (multimodal, recommended for vision)
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            # GPT-4.1 series (latest, up to 1M context)
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            # GPT-4 Turbo
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            # o-series reasoning models
            "o1",
            "o1-mini",
            "o1-preview",
            "o3",
            "o3-mini",
            "o4-mini",
        ])
        # Allow typing arbitrary model IDs (future-proof)
        self.model_combo.setEditable(True)
        # Set default from config
        idx = self.model_combo.findText(self._selected_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setCurrentText(self._selected_model)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        llm_row.addWidget(self.model_combo)

        self.llm_input = QLineEdit()
        self.llm_input.setPlaceholderText("Type an LLM command (e.g., 'move forward 25mm')…")
        llm_row.addWidget(self.llm_input, 1)
        self.llm_run_btn = QPushButton("Run LLM")
        self.llm_run_btn.clicked.connect(self._run_llm)
        llm_row.addWidget(self.llm_run_btn)
        tools_layout.addLayout(llm_row)

        # Manual tool controls
        manual_group = QGroupBox("Manual Tool Call")
        manual_layout = QVBoxLayout(manual_group)
        manual_layout.setContentsMargins(10, 10, 10, 10)
        manual_layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Tool:"))
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["move_gripper_delta", "set_gripper_percent", "go_home", "move_piece", "look_around"])
        self.tool_combo.currentTextChanged.connect(self._sync_tool_fields)
        top_row.addWidget(self.tool_combo, 1)
        manual_layout.addLayout(top_row)

        self.tool_fields = QStackedWidget()
        manual_layout.addWidget(self.tool_fields)

        # Fields: move_gripper_delta (polar interface)
        w_move = QWidget()
        wml = QVBoxLayout(w_move)
        wml.setContentsMargins(0, 0, 0, 0)
        wml.setSpacing(4)

        # Row 1: radial and angular
        row1 = QHBoxLayout()
        self.dx_spin = QDoubleSpinBox(); self.dx_spin.setRange(-200.0, 200.0); self.dx_spin.setDecimals(1); self.dx_spin.setValue(10.0)
        self.dtheta_spin = QDoubleSpinBox(); self.dtheta_spin.setRange(-180.0, 180.0); self.dtheta_spin.setDecimals(1); self.dtheta_spin.setValue(0.0)
        self.dy_spin = QDoubleSpinBox(); self.dy_spin.setRange(-200.0, 200.0); self.dy_spin.setDecimals(1); self.dy_spin.setValue(0.0)
        row1.addWidget(QLabel("Δradius(mm)")); row1.addWidget(self.dx_spin)
        row1.addWidget(QLabel("Δθ(deg)")); row1.addWidget(self.dtheta_spin)
        row1.addWidget(QLabel("dy(mm)")); row1.addWidget(self.dy_spin)
        wml.addLayout(row1)

        # Row 2: height controls
        row2 = QHBoxLayout()
        self.dz_spin = QDoubleSpinBox(); self.dz_spin.setRange(-200.0, 200.0); self.dz_spin.setDecimals(1); self.dz_spin.setValue(0.0)
        self.z_abs_spin = QDoubleSpinBox(); self.z_abs_spin.setRange(0.0, 0.6); self.z_abs_spin.setDecimals(3); self.z_abs_spin.setValue(0.0); self.z_abs_spin.setSingleStep(0.01)
        self.z_abs_check = QCheckBox("use abs Z")
        self.z_abs_check.setChecked(False)
        row2.addWidget(QLabel("Δz(mm)")); row2.addWidget(self.dz_spin)
        row2.addWidget(self.z_abs_check)
        row2.addWidget(QLabel("Z(m)")); row2.addWidget(self.z_abs_spin)
        wml.addLayout(row2)

        # Note: Large moves are automatically broken into 15mm incremental steps
        self.tool_fields.addWidget(w_move)

        # Fields: set_gripper_percent
        w_grip = QWidget()
        wgl = QHBoxLayout(w_grip)
        wgl.setContentsMargins(0, 0, 0, 0)
        self.grip_spin = QDoubleSpinBox(); self.grip_spin.setRange(0.0, 100.0); self.grip_spin.setDecimals(1); self.grip_spin.setValue(15.0)
        wgl.addWidget(QLabel("percent")); wgl.addWidget(self.grip_spin)
        wgl.addStretch()
        self.tool_fields.addWidget(w_grip)

        # Fields: go_home
        w_home = QWidget()
        whl = QHBoxLayout(w_home)
        whl.setContentsMargins(0, 0, 0, 0)
        whl.addWidget(QLabel("No args"))
        whl.addStretch()
        self.tool_fields.addWidget(w_home)

        # Fields: move_piece
        w_piece = QWidget()
        wpl = QHBoxLayout(w_piece)
        wpl.setContentsMargins(0, 0, 0, 0)
        self.from_sq = QLineEdit(); self.from_sq.setPlaceholderText("from (e2)"); self.from_sq.setText("e2")
        self.to_sq = QLineEdit(); self.to_sq.setPlaceholderText("to (e4)"); self.to_sq.setText("e4")
        self.hover_spin = QDoubleSpinBox(); self.hover_spin.setRange(0.0, 0.30); self.hover_spin.setDecimals(3); self.hover_spin.setSingleStep(0.005); self.hover_spin.setValue(0.08)
        self.transit_spin = QDoubleSpinBox(); self.transit_spin.setRange(0.0, 0.40); self.transit_spin.setDecimals(3); self.transit_spin.setSingleStep(0.005); self.transit_spin.setValue(0.15)
        wpl.addWidget(QLabel("from")); wpl.addWidget(self.from_sq)
        wpl.addWidget(QLabel("to")); wpl.addWidget(self.to_sq)
        wpl.addWidget(QLabel("hover(m)")); wpl.addWidget(self.hover_spin)
        wpl.addWidget(QLabel("transit(m)")); wpl.addWidget(self.transit_spin)
        self.tool_fields.addWidget(w_piece)

        # Fields: look_around
        w_look = QWidget()
        wll = QHBoxLayout(w_look)
        wll.setContentsMargins(0, 0, 0, 0)
        self.look_dir = QComboBox()
        self.look_dir.addItems(["left", "right", "up", "down"])
        self.look_step = QDoubleSpinBox(); self.look_step.setRange(1.0, 200.0); self.look_step.setDecimals(1); self.look_step.setSingleStep(5.0); self.look_step.setValue(25.0)
        wll.addWidget(QLabel("direction")); wll.addWidget(self.look_dir)
        wll.addWidget(QLabel("step(mm)")); wll.addWidget(self.look_step)
        wll.addStretch()
        self.tool_fields.addWidget(w_look)

        self.exec_btn = QPushButton("Execute")
        self.exec_btn.clicked.connect(self._run_manual_tool)
        manual_layout.addWidget(self.exec_btn)

        tools_layout.addWidget(manual_group)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("font-family: Menlo, Monaco, Consolas, monospace; font-size: 10pt;")
        self.log_view.setPlaceholderText("Tool calls + results will appear here…")
        tools_layout.addWidget(self.log_view, 1)

        splitter.addWidget(tools_group)
        splitter.setSizes([750, 530])

        self._sync_tool_fields(self.tool_combo.currentText())

        # Disable LLM button if no client
        if self._llm_client is None:
            self.llm_run_btn.setEnabled(False)
            if not _OPENAI_AVAILABLE:
                reason = "OpenAI client unavailable: python package 'openai' is not installed."
            elif (self.cfg.api_key or os.getenv("OPENAI_API_KEY")) is None:
                reason = "OpenAI client unavailable: set OPENAI_API_KEY (or pass --api-key)."
            elif self._llm_init_error:
                reason = f"OpenAI client failed to initialize: {self._llm_init_error}"
            else:
                reason = "OpenAI client unavailable (unknown reason)."
            self.llm_run_btn.setToolTip(reason)
            self._log(reason)

    def _append_log(self, line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"{ts} {line}")

    def _log(self, line: str) -> None:
        self.append_log_signal.emit(str(line))

    def _refresh_status(self) -> None:
        robot_ok = bool(self.tools.robot is not None and self.tools.robot.is_connected)
        kin_ok = bool(self.tools.kin is not None)
        board_ok = bool(self.tools.board_model is not None and self.tools.board_model.T_base_board is not None)
        llm_ok = bool(self._llm_client is not None)
        if llm_ok:
            llm_status = "ok"
        elif not _OPENAI_AVAILABLE:
            llm_status = "missing openai pkg"
        elif (self.cfg.api_key or os.getenv("OPENAI_API_KEY")) is None:
            llm_status = "missing API key"
        elif self._llm_init_error:
            llm_status = "init failed"
        else:
            llm_status = "unavailable"

        self.status_label.setText(
            f"Robot: {'connected' if robot_ok else 'disconnected'}"
            f" | Kinematics: {'ok' if kin_ok else 'missing'}"
            f" | Board: {'ok' if board_ok else 'missing T_base_board'}"
            f" | LLM: {llm_status}"
            f" | Torque: {'OFF' if bool(getattr(self.tools, 'torque_disabled', False)) else 'ON'}"
        )

    # -----------------------------
    # File watcher for code changes
    # -----------------------------

    def _setup_file_watcher(self) -> None:
        """Watch Python files for changes and show restart banner."""
        self._file_watcher = QFileSystemWatcher(self)
        self._file_watcher.fileChanged.connect(self._on_file_changed)
        self._file_watcher.directoryChanged.connect(self._on_directory_changed)

        # Watch the main script
        main_script = Path(__file__).resolve()
        if main_script.exists():
            self._file_watcher.addPath(str(main_script))

        # Watch the llm-tools directory
        llm_tools_dir = _LLM_TOOLS_DIR
        if llm_tools_dir.exists():
            self._file_watcher.addPath(str(llm_tools_dir))
            for py_file in llm_tools_dir.glob("*.py"):
                self._file_watcher.addPath(str(py_file))

    def _on_file_changed(self, path: str) -> None:
        """Called when a watched file changes."""
        self._log(f"File changed: {Path(path).name}")
        self.restart_banner.setVisible(True)
        # Re-add the file to the watcher (some editors replace files)
        if Path(path).exists():
            self._file_watcher.addPath(path)

    def _on_directory_changed(self, path: str) -> None:
        """Called when a watched directory changes."""
        self._log(f"Directory changed: {Path(path).name}")
        self.restart_banner.setVisible(True)
        # Re-add any new .py files
        dir_path = Path(path)
        if dir_path.exists():
            for py_file in dir_path.glob("*.py"):
                if str(py_file) not in self._file_watcher.files():
                    self._file_watcher.addPath(str(py_file))

    def _restart_app(self) -> None:
        """Restart the application to pick up code changes."""
        self._log("Restarting application...")
        # Give a moment for the log to appear
        QTimer.singleShot(100, self._do_restart)

    def _do_restart(self) -> None:
        """Actually perform the restart."""
        # Clean up resources
        try:
            if self._camera_timer:
                self._camera_timer.stop()
        except Exception:
            pass
        try:
            if self._camera is not None and self._camera.is_connected:
                self._camera.disconnect()
        except Exception:
            pass
        try:
            self.tools.disconnect_robot()
        except Exception:
            pass

        # Restart the process
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    def _sync_tool_fields(self, tool_name: str) -> None:
        name = str(tool_name)
        idx = {"move_gripper_delta": 0, "set_gripper_percent": 1, "go_home": 2, "move_piece": 3, "look_around": 4}.get(name, 0)
        self.tool_fields.setCurrentIndex(int(idx))

    # -----------------------------
    # Camera
    # -----------------------------

    def _setup_camera(self) -> None:
        try:
            cfg = OpenCVCameraConfig(
                index_or_path=int(self.cfg.camera_index),
                width=int(self.cfg.camera_width),
                height=int(self.cfg.camera_height),
                fps=int(self.cfg.camera_fps),
                color_mode=ColorMode.BGR,
            )
            self._camera = OpenCVCamera(cfg)
            self._camera.connect(warmup=True)
            self._log(f"Camera connected (index={self.cfg.camera_index})")
        except Exception as e:
            self._camera = None
            self._log(f"Camera failed to connect: {e}")

    def _tick_camera(self) -> None:
        cam = self._camera
        if cam is None or not cam.is_connected:
            self.camera_status.setText("Camera: disconnected")
            return

        try:
            frame = cam.async_read(timeout_ms=50)
            # Frame is BGR
            self._last_frame_bgr = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            bytes_per_line = 3 * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.camera_label.setPixmap(pix.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.camera_status.setText(f"Camera: {w}x{h} @ ~{self.cfg.camera_fps}fps")
        except Exception:
            # Don't spam; just keep last frame
            self.camera_status.setText("Camera: read failed")

    def capture_frame_base64(self) -> str | None:
        """Capture current camera frame as base64 JPEG for LLM vision."""
        if self._last_frame_bgr is None:
            return None
        try:
            _, buffer = cv2.imencode('.jpg', self._last_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception:
            return None

    # -----------------------------
    # Tool execution
    # -----------------------------

    def _on_model_changed(self, model: str) -> None:
        self._selected_model = str(model)
        self._log(f"LLM model changed to: {model}")

    def _set_busy(self, busy: bool) -> None:
        self.exec_btn.setEnabled(not busy)
        self.tool_combo.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.llm_run_btn.setEnabled((not busy) and (self._llm_client is not None))
        self.llm_input.setEnabled(not busy)
        # Safety: torque buttons should always be available.
        self.torque_off_btn.setEnabled(True)
        self.torque_on_btn.setEnabled(True)
        self.diag_btn.setEnabled(True)

    def _run_manual_tool(self) -> None:
        if self._tool_thread is not None and self._tool_thread.isRunning():
            return
        if bool(getattr(self.tools, "torque_disabled", False)):
            self._log("Torque is OFF. Click 'Torque ON' before running tools.")
            return

        tool = str(self.tool_combo.currentText())
        args: dict[str, Any] = {}

        if tool == "move_gripper_delta":
            args = {
                "dx_mm": float(self.dx_spin.value()),
                "dy_mm": float(self.dy_spin.value()),
                "dz_mm": float(self.dz_spin.value()),
            }
            # Add optional dtheta_deg if non-zero
            if abs(float(self.dtheta_spin.value())) > 0.01:
                args["dtheta_deg"] = float(self.dtheta_spin.value())
            # Add absolute Z if checkbox is checked
            if self.z_abs_check.isChecked():
                args["z_m"] = float(self.z_abs_spin.value())
        elif tool == "set_gripper_percent":
            args = {"percent": float(self.grip_spin.value())}
        elif tool == "go_home":
            args = {}
        elif tool == "move_piece":
            args = {
                "from_square": str(self.from_sq.text()).strip(),
                "to_square": str(self.to_sq.text()).strip(),
                "hover_height_m": float(self.hover_spin.value()),
                "transit_height_m": float(self.transit_spin.value()),
            }
        elif tool == "look_around":
            args = {
                "direction": str(self.look_dir.currentText()),
                "step_mm": float(self.look_step.value()),
            }

        self._set_busy(True)
        self._refresh_status()

        t = ToolExecutionThread(self.tools, tool, args)
        self._tool_thread = t
        t.log_line.connect(self._log)
        t.finished_ok.connect(lambda ok: (self._set_busy(False), self._refresh_status()))
        t.start()

    def _run_llm(self) -> None:
        if self._llm_client is None:
            return
        if self._llm_thread is not None and self._llm_thread.isRunning():
            return
        if bool(getattr(self.tools, "torque_disabled", False)):
            self._log("Torque is OFF. Click 'Torque ON' before running LLM tool calls.")
            return

        cmd = str(self.llm_input.text()).strip()
        if not cmd:
            return

        self._set_busy(True)
        self._refresh_status()
        self._log(f"LLM command: {cmd}")

        t = LLMThread(ui=self, llm_client=self._llm_client, model=str(self._selected_model), tools=self.tools, command=cmd)
        self._llm_thread = t
        t.log_line.connect(self._log)
        t.finished_ok.connect(lambda ok: (self._set_busy(False), self._refresh_status()))
        t.start()

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            if self._camera_timer:
                self._camera_timer.stop()
        except Exception:
            pass

        try:
            if self._camera is not None and self._camera.is_connected:
                self._camera.disconnect()
        except Exception:
            pass

        try:
            self.tools.disconnect_robot()
        except Exception:
            pass

        super().closeEvent(event)

    def _torque_off(self) -> None:
        try:
            res = self.tools.disable_torque()
            self._log(json.dumps(res, indent=2))
            if bool(res.get("ok", False)):
                self._log("TORQUE OFF: motors are now limp (E-stop).")
        except Exception as e:
            self._log(f"TORQUE OFF failed: {e}")
        self._refresh_status()

    def _torque_on(self) -> None:
        try:
            res = self.tools.enable_torque()
            self._log(json.dumps(res, indent=2))
        except Exception as e:
            self._log(f"TORQUE ON failed: {e}")
        self._refresh_status()

    def _read_motor_diagnostics(self) -> None:
        try:
            diag = self.tools.read_motor_diagnostics()
            self._log(json.dumps(diag, indent=2))
        except Exception as e:
            self._log(f"Diagnostics failed: {e}")


def _parse_args() -> AppConfig:
    import argparse

    p = argparse.ArgumentParser(description="Chess Robot UI (LLM v2): camera + tool calls")
    p.add_argument("--port", required=False, help="Robot serial port (SO-101)")
    p.add_argument("--robot-id", default="so101_chess", help="Calibration id (default: so101_chess)")
    p.add_argument("--urdf", default=None, help="URDF path (or set SO101_URDF)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.2"))
    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")

    a = p.parse_args()
    return AppConfig(
        port=a.port,
        robot_id=str(a.robot_id),
        urdf_path=str(a.urdf) if a.urdf else None,
        camera_index=int(a.camera_index),
        camera_width=int(a.camera_width),
        camera_height=int(a.camera_height),
        camera_fps=int(a.camera_fps),
        model=str(a.model),
        api_key=str(a.api_key) if a.api_key else None,
    )


def main() -> None:
    cfg = _parse_args()

    app = QApplication([])
    ui = ChessRobotUILLMV2(cfg)
    ui.show()
    app.exec()


if __name__ == "__main__":
    main()
