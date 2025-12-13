#!/usr/bin/env python

"""
Chess Robot Monitoring UI with LLM Control
Shows live camera feed, motor status, chess board diagram, and LLM-based natural language control.
"""

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QGroupBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QFrame, QToolTip, QScrollArea,
    QTextEdit, QLineEdit, QPlainTextEdit, QSizePolicy, QCheckBox, QSpinBox,
    QDialog, QSlider, QDoubleSpinBox, QListWidget, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize, QFileSystemWatcher
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont, QKeySequence, QShortcut
import cv2
import json
import time
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path
from PIL import Image
import base64
import io
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Optional, Dict, Any, List, Tuple

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    # Try to manually load .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Loaded .env file (manual)")
    else:
        print("ℹ️ No .env file found")

# LLM imports
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ OpenAI library not found. Install with: pip install openai")

# ChatKit Python SDK imports
try:
    from chatkit import ChatKitServer
    CHATKIT_SDK_AVAILABLE = True
except ImportError:
    CHATKIT_SDK_AVAILABLE = False
    # Note: ChatKit Python SDK is optional - install with: pip install openai-chatkit

# Robot imports
from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.model.kinematics import RobotKinematics

# Qt styling libraries
try:
    import qdarkstyle
    QDARKSTYLE_AVAILABLE = True
except ImportError:
    QDARKSTYLE_AVAILABLE = False
    print("ℹ️ qdarkstyle not available - install with: pip install qdarkstyle")

try:
    from lerobot.ui.style_system import StyleSystem, btn_purple, btn_primary, btn_success, btn_danger, input_dark, badge_info, badge_success, badge_warning, badge_danger, card_dark
    STYLE_SYSTEM_AVAILABLE = True
except ImportError:
    STYLE_SYSTEM_AVAILABLE = False


class ChessBoardWidget(QWidget):
    """Custom widget for drawing chess board."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(260, 260)
        self.setStyleSheet("background-color: #2c3e50;")
        self._board_drawn = False
        self.robot_square = None
        
    def paintEvent(self, event):
        """Draw chess board with optional robot position highlight."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        square_size = 30
        
        # Draw 8x8 grid
        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Alternating colors
                if (row + col) % 2 == 0:
                    color = QColor('#f8f9fa')
                    text_color = QColor('#343a40')
                else:
                    color = QColor('#6c757d')
                    text_color = QColor('#f8f9fa')
                
                # Draw square
                painter.fillRect(x1, y1, square_size, square_size, color)
                painter.setPen(QPen(QColor('#343a40'), 1))
                painter.drawRect(x1, y1, square_size, square_size)
                
                # Add square label
                square = chr(ord('a') + col) + str(8 - row)
                painter.setPen(QPen(text_color))
                font = QFont("Arial", 7, QFont.Bold)
                painter.setFont(font)
                painter.drawText(x1 + 15, y1 + 8, square)
        
        # Add board edge labels
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor('#ecf0f1')))
        
        # File labels (a-h) at bottom
        for col in range(8):
            x = col * square_size + 15
            painter.drawText(x, 245, chr(ord('a') + col))
        
        # Rank labels (1-8) on right side
        for row in range(8):
            y = row * square_size + 15
            painter.drawText(245, y, str(8 - row))
        
        # Highlight robot position if set
        if self.robot_square:
            self._draw_robot_highlight(painter, self.robot_square)
        
        self._board_drawn = True
    
    def _draw_robot_highlight(self, painter, square):
        """Draw robot position highlight on chess board."""
        file_idx = ord(square[0]) - ord('a')
        rank_idx = int(square[1]) - 1
        display_row = 7 - rank_idx
        square_size = 30
        
        x1 = file_idx * square_size
        y1 = display_row * square_size
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        # Highlight border
        painter.setPen(QPen(QColor('#e74c3c'), 4))
        painter.drawRect(x1 + 1, y1 + 1, square_size - 2, square_size - 2)
        
        # Inner glow
        painter.setPen(QPen(QColor('#ff6b6b'), 2))
        painter.drawRect(x1 + 3, y1 + 3, square_size - 6, square_size - 6)
        
        # Robot icon
        font = QFont("Arial", 14, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor('#e74c3c')))
        painter.drawText(x1 + 15, y1 + 15, "🤖")
        
        # Square label
        font = QFont("Arial", 8, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor('#ffffff')))
        painter.drawText(x1 + 15, y1 + 25, square.upper())
        
        # Crosshairs
        center_x, center_y = x1 + 15, y1 + 15
        painter.setPen(QPen(QColor('#e74c3c'), 2))
        painter.drawLine(center_x, y1, center_x, y2)
        painter.drawLine(x1, center_y, x2, center_y)
        
        # Direction indicators
        painter.setPen(QPen(QColor('#3498db'), 1))
        for angle in [0, 90, 180, 270]:
            rad = np.radians(angle)
            end_x = center_x + 8 * np.cos(rad)
            end_y = center_y + 8 * np.sin(rad)
            painter.drawLine(center_x, center_y, int(end_x), int(end_y))
    
    def set_robot_square(self, square):
        """Set robot square and trigger repaint."""
        self.robot_square = square
        self.update()


class WorkspaceWidget(QWidget):
    """Custom widget for workspace visualization."""
    
    def __init__(self, parent=None, view_type='top'):
        super().__init__(parent)
        self.view_type = view_type
        if view_type == 'top':
            self.setFixedSize(120, 120)
        else:
            self.setFixedSize(300, 200)
        self.setStyleSheet("background-color: #2c3e50; border: 1px solid #95a5a6;")
        self.robot_pos = None  # (x, y, z) in mm
        
    def paintEvent(self, event):
        """Draw workspace visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.view_type == 'top':
            self._draw_top_view(painter)
        else:
            self._draw_side_view(painter)
    
    def _draw_top_view(self, painter):
        """Draw top view of workspace."""
        center = 60
        max_reach = 350
        min_reach = 100
        scale = 0.3
        
        # Draw robot base
        painter.setBrush(QBrush(QColor('#2c3e50')))
        painter.setPen(QPen(QColor('#ecf0f1'), 2))
        painter.drawEllipse(center - 3, center - 3, 6, 6)
        
        font = QFont("Arial", 6)
        painter.setFont(font)
        painter.setPen(QPen(QColor('#ecf0f1')))
        painter.drawText(center - 10, center - 10, "BASE")
        
        # Draw workspace boundaries
        outer_radius = max_reach * scale
        inner_radius = min_reach * scale
        mid_radius = (max_reach + min_reach) / 2 * scale
        
        # Outer boundary
        pen = QPen(QColor('#3498db'), 2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center - outer_radius, center - outer_radius, 
                           outer_radius * 2, outer_radius * 2)
        
        # Inner dead zone
        pen = QPen(QColor('#95a5a6'), 1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawEllipse(center - inner_radius, center - inner_radius,
                           inner_radius * 2, inner_radius * 2)
        
        # Optimal workspace
        painter.setPen(QPen(QColor('#27ae60'), 2))
        painter.drawEllipse(center - mid_radius, center - mid_radius,
                           mid_radius * 2, mid_radius * 2)
        
        # Draw axes
        painter.setPen(QPen(QColor('#e74c3c'), 2))
        painter.drawLine(center, center, center, center - 60)
        font = QFont("Arial", 8, QFont.Bold)
        painter.setFont(font)
        painter.drawText(center + 20, center - 50, "X")
        
        painter.setPen(QPen(QColor('#27ae60'), 2))
        painter.drawLine(center, center, center + 60, center)
        painter.drawText(center + 50, center - 15, "Y")
        
        # Draw robot position if set
        if self.robot_pos:
            x_mm, y_mm, z_mm = self.robot_pos
            canvas_x = center + y_mm * scale
            canvas_y = center - x_mm * scale
            
            canvas_x = max(10, min(110, canvas_x))
            canvas_y = max(10, min(110, canvas_y))
            
            # Color based on height
            if z_mm > 150:
                color = QColor('#27ae60')
            elif z_mm > 50:
                color = QColor('#f39c12')
            else:
                color = QColor('#e74c3c')
            
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor('#ffffff'), 2))
            painter.drawEllipse(int(canvas_x) - 5, int(canvas_y) - 5, 10, 10)
            
            font = QFont("Arial", 7)
            painter.setFont(font)
            painter.setPen(QPen(QColor('#ecf0f1')))
            painter.drawText(int(canvas_x) + 12, int(canvas_y), 
                           f"({x_mm:.0f},{y_mm:.0f})")
    
    def _draw_side_view(self, painter):
        """Draw side view of workspace."""
        center_x, center_z = 150, 100
        max_reach = 350
        min_reach = 100
        
        # Draw robot base
        painter.setBrush(QBrush(QColor('#2c3e50')))
        painter.setPen(QPen(QColor('#ecf0f1'), 2))
        painter.drawEllipse(center_x - 3, 180, 6, 6)
        
        font = QFont("Arial", 8)
        painter.setFont(font)
        painter.setPen(QPen(QColor('#ecf0f1')))
        painter.drawText(center_x, 195, "BASE")
        
        # Draw workspace envelope
        workspace_points = []
        for angle in np.linspace(-np.pi/2, np.pi/2, 20):
            x_outer = center_x + max_reach * np.cos(angle) * 0.3
            z_outer = center_z - max_reach * np.sin(angle) * 0.3
            workspace_points.append((x_outer, z_outer))
        
        if len(workspace_points) >= 3:
            from PySide6.QtCore import QPoint
            points = [QPoint(int(p[0]), int(p[1])) for p in workspace_points]
            from PySide6.QtGui import QPolygon
            poly = QPolygon(points)
            painter.setBrush(QBrush(QColor('#3498db')))
            painter.setPen(QPen(QColor('#3498db'), 2))
            painter.drawPolygon(poly)
        
        # Draw axes
        painter.setPen(QPen(QColor('#e74c3c'), 2))
        painter.drawLine(center_x, 180, center_x + 60, 180)
        painter.drawText(center_x + 70, 180, "X (forward)")
        
        painter.setPen(QPen(QColor('#3498db'), 2))
        painter.drawLine(center_x, 180, center_x, 120)
        painter.drawText(center_x + 15, 110, "Z (up)")
        
        # Draw robot position if set
        if self.robot_pos:
            x_mm, y_mm, z_mm = self.robot_pos
            scale = 0.3
            pos_x = center_x + x_mm * scale
            pos_z = center_z - z_mm * scale
            
            pos_x = max(10, min(290, pos_x))
            pos_z = max(10, min(190, pos_z))
            
            painter.setBrush(QBrush(QColor('#e74c3c')))
            painter.setPen(QPen(QColor('#ffffff'), 2))
            painter.drawEllipse(int(pos_x) - 4, int(pos_z) - 4, 8, 8)
            
            font = QFont("Arial", 7)
            painter.setFont(font)
            painter.setPen(QPen(QColor('#ecf0f1')))
            painter.drawText(int(pos_x) + 10, int(pos_z), 
                           f"({x_mm:.0f},{z_mm:.0f})")
    
    
    def set_robot_position(self, x_mm, y_mm, z_mm):
        """Set robot position and trigger repaint."""
        self.robot_pos = (x_mm, y_mm, z_mm)
        self.update()


class Robot3DWidget(QWidget):
    """3D visualization of the robot arm."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Robot link lengths (mm)
        self.L0 = 100  # Base to shoulder
        self.L1 = 150  # Shoulder to elbow
        self.L2 = 120  # Elbow to wrist
        self.L3 = 80   # Wrist to gripper
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(6, 5), facecolor='#2c3e50')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # Set up layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initialize joint positions (all zeros)
        self.joint_angles = {
            "shoulder_pan": 0,
            "shoulder_lift": 0,
            "elbow_flex": 0,
            "wrist_flex": 0,
            "wrist_roll": 0,
            "gripper": 0
        }
        
        # Draw initial robot
        self.draw_robot()
        
    def calculate_joint_positions(self):
        """Calculate 3D positions of all joints using forward kinematics."""
        # Convert angles to radians
        shoulder_pan = np.radians(self.joint_angles["shoulder_pan"])
        shoulder_lift = np.radians(self.joint_angles["shoulder_lift"])
        elbow_flex = np.radians(self.joint_angles["elbow_flex"])
        wrist_flex = np.radians(self.joint_angles["wrist_flex"])
        
        # Base position (origin)
        base_pos = np.array([0, 0, 0])
        
        # Base to shoulder (L0 height)
        shoulder_pos = np.array([0, 0, self.L0])
        
        # Shoulder to elbow (L1)
        # In the plane defined by shoulder_pan, lift by shoulder_lift
        # Forward kinematics: L1 in the rotated frame
        reach_1 = self.L1 * np.cos(shoulder_lift)
        height_1 = self.L1 * np.sin(shoulder_lift)
        
        # In the pan-rotated frame
        elbow_local = np.array([reach_1, 0, height_1])
        
        # Apply pan rotation
        R_pan = np.array([
            [np.cos(shoulder_pan), -np.sin(shoulder_pan), 0],
            [np.sin(shoulder_pan), np.cos(shoulder_pan), 0],
            [0, 0, 1]
        ])
        elbow_pos = shoulder_pos + R_pan @ elbow_local
        
        # Elbow to wrist (L2)
        # Total angle from shoulder = shoulder_lift + elbow_flex
        total_lift = shoulder_lift + elbow_flex
        reach_2 = self.L2 * np.cos(total_lift)
        height_2 = self.L2 * np.sin(total_lift)
        
        wrist_local = np.array([reach_2, 0, height_2])
        wrist_pos = elbow_pos + R_pan @ wrist_local
        
        # Wrist to gripper (L3)
        # Total angle from shoulder = total_lift + wrist_flex
        total_wrist = total_lift + wrist_flex
        reach_3 = self.L3 * np.cos(total_wrist)
        height_3 = self.L3 * np.sin(total_wrist)
        
        gripper_local = np.array([reach_3, 0, height_3])
        gripper_pos = wrist_pos + R_pan @ gripper_local
        
        return {
            "base": base_pos,
            "shoulder": shoulder_pos,
            "elbow": elbow_pos,
            "wrist": wrist_pos,
            "gripper": gripper_pos
        }
    
    def draw_robot(self):
        """Draw the robot arm in 3D."""
        self.ax.clear()
        
        # Calculate joint positions
        positions = self.calculate_joint_positions()
        
        # Define colors for each link
        colors = {
            "base": '#3498db',
            "shoulder": '#e74c3c',
            "elbow": '#27ae60',
            "wrist": '#f39c12',
            "gripper": '#9b59b6'
        }
        
        # Draw base
        base_pos = positions["base"]
        self.ax.scatter([base_pos[0]], [base_pos[1]], [base_pos[2]], 
                       c=[colors["base"]], s=100, marker='o', label='Base')
        
        # Draw links
        links = [
            (positions["base"], positions["shoulder"], colors["base"], "Base-Shoulder"),
            (positions["shoulder"], positions["elbow"], colors["shoulder"], "Shoulder-Elbow"),
            (positions["elbow"], positions["wrist"], colors["elbow"], "Elbow-Wrist"),
            (positions["wrist"], positions["gripper"], colors["wrist"], "Wrist-Gripper")
        ]
        
        for start, end, color, label in links:
            self.ax.plot([start[0], end[0]], 
                         [start[1], end[1]], 
                         [start[2], end[2]], 
                         color=color, linewidth=4, label=label)
        
        # Draw joints
        joint_colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']
        joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist', 'Gripper']
        joint_positions = [positions["base"], positions["shoulder"], 
                          positions["elbow"], positions["wrist"], positions["gripper"]]
        
        for i, (pos, color, name) in enumerate(zip(joint_positions, joint_colors, joint_names)):
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                           c=[color], s=150, marker='o')
            # Add joint labels
            self.ax.text(pos[0], pos[1], pos[2], f' {name}', fontsize=8, color=color)
        
        # Set axis limits
        max_reach = 450
        self.ax.set_xlim([-max_reach, max_reach])
        self.ax.set_ylim([-max_reach, max_reach])
        self.ax.set_zlim([0, max_reach])
        
        # Set labels
        self.ax.set_xlabel('X (mm)', color='white')
        self.ax.set_ylabel('Y (mm)', color='white')
        self.ax.set_zlabel('Z (mm)', color='white')
        
        # Set background color
        self.ax.set_facecolor('#2c3e50')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#95a5a6')
        self.ax.yaxis.pane.set_edgecolor('#95a5a6')
        self.ax.zaxis.pane.set_edgecolor('#95a5a6')
        
        # Set tick colors
        self.ax.tick_params(colors='white')
        
        # Set title
        self.ax.set_title('Robot Arm 3D View', color='white', fontsize=12, pad=10)
        
        # Enable interactive rotation
        self.ax.view_init(elev=20, azim=45)
        
        # Draw coordinate axes
        axis_length = 100
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='#e74c3c', arrow_length_ratio=0.2, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='#27ae60', arrow_length_ratio=0.2, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='#3498db', arrow_length_ratio=0.2, linewidth=2)
        
        self.canvas.draw()
    
    def update_joints(self, joint_angles):
        """Update joint angles and redraw robot."""
        self.joint_angles = joint_angles
        self.draw_robot()


class LLMExecutionThread(QThread):
    """Thread for executing LLM commands - prevents UI blocking."""
    
    status_update = Signal(str)
    reasoning_update = Signal(str)
    response_update = Signal(str)
    action_preview_update = Signal(str)
    finished_signal = Signal(bool, str)  # success, message
    
    def __init__(self, ui_instance, command: str, parent=None):
        super().__init__(parent)
        self.ui = ui_instance
        self.command = command
        self.running = True
    
    def run(self):
        """Execute LLM command in background thread."""
        try:
            # Call the actual execution method
            self.ui._execute_llm_command_internal(self.command, self)
        except Exception as e:
            self.finished_signal.emit(False, f"Execution error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop execution."""
        self.running = False


class MonitoringThread(QThread):
    """Thread for monitoring robot and cameras."""
    
    main_camera_update = Signal(object)  # QPixmap for main camera
    gripper_camera_update = Signal(object)  # QPixmap for gripper camera
    motor_update = Signal(dict)  # Motor data dict
    status_update = Signal(str)  # Status message
    
    def __init__(self, bus, cameras, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.cameras = cameras  # dict of cameras
        self.running = False
        self.paused = False  # Pause motor reads during command execution to avoid port conflicts
        
    def run(self):
        """Main monitoring loop."""
        try:
            # Connect systems
            self.bus._connect(handshake=False)
            
            # Configure motors 3, 4, and 6 with gentler settings to reduce stiffness
            print("🔧 Configuring motors 3, 4, and 6 with gentler settings...")
            with self.bus.torque_disabled():
                # Configure motor 3 (elbow_flex) - reduce stiffness
                # First verify motor 3 exists and can be read before configuring
                if "elbow_flex" in self.bus.motors:
                    try:
                        # Verify motor exists by trying to read its position
                        test_read = self.bus.read("Present_Position", "elbow_flex", normalize=False)
                        print(f"  📝 Configuring elbow_flex (motor 3) - current position: {test_read}")
                        
                        self.bus.write("Operating_Mode", "elbow_flex", OperatingMode.POSITION.value)
                        # Reduce P coefficient for less stiffness (default is 32, we use 12)
                        self.bus.write("P_Coefficient", "elbow_flex", 12)
                        # Reduce D coefficient (default is 32, we use 18)
                        self.bus.write("D_Coefficient", "elbow_flex", 18)
                        # Keep I coefficient at 0
                        self.bus.write("I_Coefficient", "elbow_flex", 0)
                        # Add torque limits to prevent overload
                        self.bus.write("Max_Torque_Limit", "elbow_flex", 650)  # 65% of max (slightly higher than wrist)
                        self.bus.write("Protection_Current", "elbow_flex", 325)  # 65% of max current
                        self.bus.write("Overload_Torque", "elbow_flex", 32)  # 32% torque when overloaded
                        print("    ✅ elbow_flex configured with reduced stiffness")
                    except Exception as e:
                        print(f"    ⚠️ Could not configure elbow_flex (motor 3): {e}")
                        print(f"    ⏭️ Skipping motor 3 configuration - motor may not be connected or detected")
                
                # Configure motor 4 (wrist_flex) - reduce stiffness
                if "wrist_flex" in self.bus.motors:
                    try:
                        # Verify motor exists
                        test_read = self.bus.read("Present_Position", "wrist_flex", normalize=False)
                        print(f"  📝 Configuring wrist_flex (motor 4) - current position: {test_read}")
                        
                        self.bus.write("Operating_Mode", "wrist_flex", OperatingMode.POSITION.value)
                        # Reduce P coefficient for less stiffness (default is 32, we use 10)
                        self.bus.write("P_Coefficient", "wrist_flex", 10)
                        # Reduce D coefficient (default is 32, we use 16)
                        self.bus.write("D_Coefficient", "wrist_flex", 16)
                        # Keep I coefficient at 0
                        self.bus.write("I_Coefficient", "wrist_flex", 0)
                        # Add torque limits similar to gripper to prevent overload
                        self.bus.write("Max_Torque_Limit", "wrist_flex", 600)  # 60% of max
                        self.bus.write("Protection_Current", "wrist_flex", 300)  # 60% of max current
                        self.bus.write("Overload_Torque", "wrist_flex", 30)  # 30% torque when overloaded
                        print("    ✅ wrist_flex configured with reduced stiffness")
                    except Exception as e:
                        print(f"    ⚠️ Could not configure wrist_flex (motor 4): {e}")
                
                # Configure motor 6 (gripper) - further reduce stiffness
                if "gripper" in self.bus.motors:
                    try:
                        # Verify motor exists
                        test_read = self.bus.read("Present_Position", "gripper", normalize=False)
                        print(f"  📝 Configuring gripper (motor 6) - current position: {test_read}")
                        
                        self.bus.write("Operating_Mode", "gripper", OperatingMode.POSITION.value)
                        # Reduce P coefficient further (default is 32, we use 8)
                        self.bus.write("P_Coefficient", "gripper", 8)
                        # Reduce D coefficient (default is 32, we use 12)
                        self.bus.write("D_Coefficient", "gripper", 12)
                        # Keep I coefficient at 0
                        self.bus.write("I_Coefficient", "gripper", 0)
                        # Reduce torque limits further to prevent overload
                        self.bus.write("Max_Torque_Limit", "gripper", 400)  # 40% of max (reduced from 500)
                        self.bus.write("Protection_Current", "gripper", 200)  # 40% of max current (reduced from 250)
                        self.bus.write("Overload_Torque", "gripper", 20)  # 20% torque when overloaded (reduced from 25)
                        print("    ✅ gripper configured with reduced stiffness")
                    except Exception as e:
                        print(f"    ⚠️ Could not configure gripper (motor 6): {e}")
            
            print("✅ Motor configuration complete")
            
            # Connect all cameras
            for cam_name, camera in self.cameras.items():
                try:
                    camera.connect()
                    print(f"✅ {cam_name} camera connected")
                except Exception as e:
                    print(f"⚠️ {cam_name} camera connection failed: {e}")
            
            self.status_update.emit("✅ Robot and cameras connected - Live monitoring active")
            
            # Performance optimization: separate update counters
            iteration = 0
            motor_update_counter = 0
            # Track motors that are having issues to skip them temporarily
            motor_skip_count = {}  # motor_name -> skip countdown
            
            while self.running:
                iteration += 1
                
                # Update gripper camera (every iteration for smooth video)
                try:
                    gripper_cam = self.cameras.get("gripper")
                    if gripper_cam and gripper_cam.is_connected:
                        frame = gripper_cam.read()
                        if frame is not None and frame.size > 0:
                            # Convert BGR to RGB for Qt display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_resized = cv2.resize(frame_rgb, (640, 480))  # Larger display
                            
                            height, width, channel = frame_resized.shape
                            bytes_per_line = 3 * width
                            q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
                            pixmap = QPixmap.fromImage(q_image)
                            self.gripper_camera_update.emit(pixmap)
                except Exception as e:
                    pass
                
                # Update motors - only every 3rd iteration (reduces bus traffic)
                # Skip motor reads when paused (during command execution)
                motor_update_counter += 1
                if motor_update_counter >= 3 and not self.paused:
                    motor_update_counter = 0
                    try:
                        if self.bus.is_connected:
                            motor_data = {}
                            for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                                             "wrist_flex", "wrist_roll", "gripper"]:
                                # Skip motors that are having issues (overload, etc.)
                                if motor_name in motor_skip_count:
                                    motor_skip_count[motor_name] -= 1
                                    if motor_skip_count[motor_name] <= 0:
                                        del motor_skip_count[motor_name]
                                    motor_data[motor_name] = {"position": None, "status": "skipped"}
                                    continue
                                
                                try:
                                    # Read position with error handling - use non-blocking read
                                    try:
                                        pos = self.bus.read("Present_Position", motor_name, normalize=True, num_retry=0)
                                        signals = {"position": pos, "status": "ok"}
                                    except (RuntimeError, ConnectionError) as e:
                                        error_msg = str(e).lower()
                                        # Check for overload or other motor errors
                                        if "overload" in error_msg or "error" in error_msg:
                                            # Skip this motor for 10 iterations (~1 second)
                                            motor_skip_count[motor_name] = 10
                                            motor_data[motor_name] = {"position": None, "status": "overload"}
                                            continue
                                        else:
                                            # Other error - mark but don't skip
                                            signals = {"position": None, "status": "error"}
                                    
                                    # Only read additional signals if position read succeeded
                                    if signals.get("status") == "ok":
                                        # Read additional signals less frequently (only critical ones)
                                        try:
                                            signals["velocity"] = self.bus.read("Present_Velocity", motor_name, normalize=False, num_retry=0)
                                        except:
                                            signals["velocity"] = None
                                        
                                        try:
                                            signals["voltage"] = self.bus.read("Present_Voltage", motor_name, normalize=False, num_retry=0)
                                        except:
                                            signals["voltage"] = None
                                        
                                        try:
                                            signals["moving"] = self.bus.read("Moving", motor_name, normalize=False, num_retry=0)
                                        except:
                                            signals["moving"] = None
                                        
                                        # Optional signals (read less frequently or skip)
                                        signals["load"] = None
                                        signals["temperature"] = None
                                        signals["current"] = None
                                        signals["goal_position"] = None
                                        signals["torque_enable"] = None
                                    
                                    motor_data[motor_name] = signals
                                except Exception as e:
                                    # Catch-all for any other errors
                                    error_msg = str(e).lower()
                                    if "overload" in error_msg:
                                        motor_skip_count[motor_name] = 10
                                    motor_data[motor_name] = {"position": None, "status": "error"}
                            self.motor_update.emit(motor_data)
                    except Exception as e:
                        # Don't let bus-level errors crash the monitoring loop
                        pass
                
                # Faster camera updates, slower motor updates
                time.sleep(0.033)  # ~30 FPS for cameras, ~10 FPS effective for motors
                
        except Exception as e:
            self.status_update.emit(f"❌ Monitoring error: {e}")
        finally:
            try:
                if hasattr(self, 'bus'):
                    self.bus.disconnect()
                if hasattr(self, 'cameras'):
                    for camera in self.cameras.values():
                        try:
                            camera.disconnect()
                        except:
                            pass
            except:
                pass
    
    def stop(self):
        """Stop monitoring."""
        self.running = False


class ChessRobotUILLM(QMainWindow):
    # Thread-safe UI update signals (queued to the main Qt thread)
    exec_log_line_signal = Signal(str)
    exec_log_reset_signal = Signal()
    execution_ui_mode_signal = Signal(bool)

    def __init__(self, port: Optional[str] = None, dev_mode: bool = False, api_key: Optional[str] = None, workflow_id: Optional[str] = None):
        super().__init__()
        self.port = port
        self.running = False
        self.dev_mode = dev_mode
        self.file_watcher = None
        
        # Setup LLM with ChatKit support
        self.setup_llm(api_key, workflow_id)
        
        # Setup robot and camera
        self.setup_robot()
        self.setup_camera()
        self.setup_kinematics()
        
        # Internal position model - tracks current robot state
        self.position_model = {
            "joints": {},  # Current joint positions
            "end_effector": None,  # Current end-effector position (x, y, z)
            "gripper": 0.0,  # Current gripper state
            "last_update": None  # Timestamp of last update
        }
        self.update_position_model()  # Initialize with current positions

        # Execution log (ring buffer) - used for UI visibility + LLM context during loops
        self._exec_log_lines = deque(maxlen=400)  # pre-formatted lines
        self._exec_log_prompt_max_lines = 30
        self._execution_active = False
        self._exec_log_display_max_lines = 250
        
        # Workspace estimate (computed from calibration + FK sampling)
        self.workspace_estimate = self._load_workspace_estimate()
        if not self.workspace_estimate:
            # Safe: computation only (no motor movement)
            self.workspace_estimate = self._estimate_workspace_from_calibration(sample_count=1200)
            self._save_workspace_estimate(self.workspace_estimate)

        # Vision+motor localization hints for "where the board is in view"
        self.board_view_calibration = self._load_board_view_calibration() or {}
        
        # Create main window - adaptive sizing
        self.setWindowTitle("Chess Robot Monitor - LLM Control Interface")
        
        # Get screen size and use a reasonable fraction
        try:
            screen = QApplication.primaryScreen().geometry()
            screen_width = screen.width()
            screen_height = screen.height()
        except:
            # Fallback if screen size can't be determined
            screen_width = 1440
            screen_height = 900
        
        # Use 90% of screen size, but cap at reasonable max
        window_width = min(int(screen_width * 0.9), 1600)
        window_height = min(int(screen_height * 0.9), 1000)
        
        # Ensure minimum size
        window_width = max(window_width, 1200)
        window_height = max(window_height, 800)
        
        # Center the window
        x_pos = max(0, (screen_width - window_width) // 2)
        y_pos = max(0, (screen_height - window_height) // 2)
        
        self.setGeometry(x_pos, y_pos, window_width, window_height)
        
        # Make window resizable with minimum size
        self.setMinimumSize(1200, 800)
        
        # Terminal-like minimal styling
        self.setStyleSheet("""
            QMainWindow {
                background: #0d1117;
                color: #c9d1d9;
            }
            QWidget {
                background: #0d1117;
                color: #c9d1d9;
                font-family: 'Monaco', 'Menlo', 'Consolas', 'Courier New', monospace;
                font-size: 11pt;
            }
            QGroupBox {
                background: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 0px;
                padding-top: 15px;
                padding-bottom: 10px;
                padding-left: 10px;
                padding-right: 10px;
                margin-top: 5px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #58a6ff;
            }
            QPushButton {
                background: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 0px;
                padding: 8px 16px;
                font-size: 11pt;
                font-weight: normal;
            }
            QPushButton:hover {
                background: #30363d;
                border: 1px solid #58a6ff;
            }
            QPushButton:pressed {
                background: #161b22;
            }
            QPushButton:disabled {
                background: #161b22;
                color: #6e7681;
                border: 1px solid #21262d;
            }
            QLabel {
                background: transparent;
                color: #c9d1d9;
                border: none;
            }
            QLineEdit, QPlainTextEdit, QTextEdit {
                background: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 0px;
                padding: 6px;
                font-family: 'Monaco', 'Menlo', 'Consolas', 'Courier New', monospace;
                font-size: 11pt;
            }
            QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {
                border: 1px solid #58a6ff;
                background: #161b22;
            }
            QComboBox {
                background: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 0px;
                padding: 6px;
                font-size: 11pt;
            }
            QComboBox:hover {
                border: 1px solid #58a6ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
                selection-background-color: #1f6feb;
            }
            QScrollArea {
                border: 1px solid #30363d;
                background: #0d1117;
            }
            QScrollBar:vertical {
                background: #0d1117;
                width: 12px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                min-height: 20px;
                border-radius: 0px;
            }
            QScrollBar::handle:vertical:hover {
                background: #484f58;
            }
            QFrame {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 0px;
            }
            QToolTip {
                background-color: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 0px;
                padding: 4px 8px;
                font-size: 10pt;
            }
        """)
        print("✅ Applied terminal-like minimal theme")
        
        # Initialize UI components
        self.create_widgets()
        
        # Setup file watcher for hot reload in dev mode
        if self.dev_mode:
            self.setup_file_watcher()
        
        # Start monitoring
        self.start_monitoring()

    def _workspace_estimate_path(self) -> Path:
        calib_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
        calib_dir.mkdir(parents=True, exist_ok=True)
        return calib_dir / "workspace_estimate.json"

    def _board_view_calibration_path(self) -> Path:
        calib_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
        calib_dir.mkdir(parents=True, exist_ok=True)
        return calib_dir / "board_view_calibration.json"

    def _load_board_view_calibration(self) -> Optional[Dict[str, Any]]:
        try:
            p = self._board_view_calibration_path()
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _save_board_view_calibration(self, calib: Optional[Dict[str, Any]]):
        try:
            if not calib:
                return
            p = self._board_view_calibration_path()
            with open(p, "w") as f:
                json.dump(calib, f, indent=2)
        except Exception:
            pass

    # NOTE: We intentionally do NOT use OpenCV chessboard detectors for localization.
    # Detection and reasoning are performed via LLM vision to be robust to non-standard boards,
    # lighting, and partial views.

    def _load_workspace_estimate(self) -> Optional[Dict[str, Any]]:
        try:
            p = self._workspace_estimate_path()
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _save_workspace_estimate(self, estimate: Optional[Dict[str, Any]]):
        try:
            if not estimate:
                return
            p = self._workspace_estimate_path()
            with open(p, "w") as f:
                json.dump(estimate, f, indent=2)
        except Exception:
            pass

    def _joint_limits_deg_from_calibration(self) -> Dict[str, Dict[str, float]]:
        """Derive per-joint min/max in degrees from motor calibration range_min/max.

        In lerobot's DEGREES mode, degrees are computed as: (raw - mid) * 360 / (resolution-1),
        where mid = (range_min + range_max)/2. So min/max degrees come directly from raw min/max.
        """
        limits: Dict[str, Dict[str, float]] = {}
        try:
            if not getattr(self, "bus", None) or not getattr(self.bus, "calibration", None):
                return limits
            for motor_name, cal in self.bus.calibration.items():
                if motor_name not in self.all_motors:
                    continue
                if self.all_motors[motor_name].norm_mode is not MotorNormMode.DEGREES:
                    continue
                min_raw = float(cal.range_min)
                max_raw = float(cal.range_max)
                mid = (min_raw + max_raw) / 2.0
                max_res = float(self.bus.model_resolution_table[self.all_motors[motor_name].model] - 1)
                min_deg = (min_raw - mid) * 360.0 / max_res
                max_deg = (max_raw - mid) * 360.0 / max_res
                lo = float(min(min_deg, max_deg))
                hi = float(max(min_deg, max_deg))
                limits[motor_name] = {"min": lo, "max": hi}
        except Exception as e:
            self._append_exec_log("warning", f"Workspace: failed to derive joint limits from calibration: {e}")
        return limits

    def _estimate_workspace_from_calibration(self, sample_count: int = 1200) -> Dict[str, Any]:
        """Estimate reachable XYZ bounds by sampling joint angles within calibration limits and running FK.

        This does NOT move motors; it uses FK + saved calibration limits only.
        """
        estimate: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "method": "fk_sampling_from_calibration",
            "sample_count": int(sample_count),
            "joint_limits_deg": {},
            "ee_bounds_mm": {},
            "ee_percentiles_mm": {},
        }
        if self.kinematics is None:
            estimate["error"] = "kinematics_unavailable"
            return estimate

        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        limits = self._joint_limits_deg_from_calibration()
        # Fallback conservative defaults if calibration missing
        fallback = {
            "shoulder_pan": (-90.0, 90.0),
            "shoulder_lift": (-90.0, 90.0),
            "elbow_flex": (-90.0, 90.0),
            "wrist_flex": (-90.0, 90.0),
            "wrist_roll": (-170.0, 170.0),
        }
        for j in joint_names:
            lo, hi = fallback[j]
            if j in limits:
                lo = float(limits[j]["min"])
                hi = float(limits[j]["max"])
            estimate["joint_limits_deg"][j] = {"min": lo, "max": hi}

        # Sample
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []

        rng = np.random.default_rng(0)
        for _ in range(int(sample_count)):
            q = []
            for j in joint_names:
                lo = estimate["joint_limits_deg"][j]["min"]
                hi = estimate["joint_limits_deg"][j]["max"]
                q.append(float(rng.uniform(lo, hi)))
            try:
                T = self.kinematics.forward_kinematics(np.array(q, dtype=float))
                xs.append(float(T[0, 3] * 1000.0))
                ys.append(float(T[1, 3] * 1000.0))
                zs.append(float(T[2, 3] * 1000.0))
            except Exception:
                continue

        if not xs:
            estimate["error"] = "fk_failed_no_samples"
            return estimate

        arr = np.array([xs, ys, zs], dtype=float)
        mins = arr.min(axis=1)
        maxs = arr.max(axis=1)
        p5 = np.percentile(arr, 5, axis=1)
        p95 = np.percentile(arr, 95, axis=1)

        estimate["ee_bounds_mm"] = {
            "x_min": float(mins[0]),
            "x_max": float(maxs[0]),
            "y_min": float(mins[1]),
            "y_max": float(maxs[1]),
            "z_min": float(mins[2]),
            "z_max": float(maxs[2]),
        }
        estimate["ee_percentiles_mm"] = {
            "x_p5": float(p5[0]),
            "x_p95": float(p95[0]),
            "y_p5": float(p5[1]),
            "y_p95": float(p95[1]),
            "z_p5": float(p5[2]),
            "z_p95": float(p95[2]),
        }

        # Recommend a stable "scan posture" near mid-range (useful for startup sweeps)
        rec = {}
        for j in joint_names:
            lo = estimate["joint_limits_deg"][j]["min"]
            hi = estimate["joint_limits_deg"][j]["max"]
            rec[j] = float((lo + hi) / 2.0)
        # Encourage a downward-looking camera by biasing wrist_flex negative if allowed
        wf_lo = estimate["joint_limits_deg"]["wrist_flex"]["min"]
        wf_hi = estimate["joint_limits_deg"]["wrist_flex"]["max"]
        rec["wrist_flex"] = float(np.clip(-30.0, wf_lo, wf_hi))
        wr_lo = estimate["joint_limits_deg"]["wrist_roll"]["min"]
        wr_hi = estimate["joint_limits_deg"]["wrist_roll"]["max"]
        rec["wrist_roll"] = float(np.clip(0.0, wr_lo, wr_hi))
        estimate["recommended_scan_pose_deg"] = rec

        return estimate
    
    def setup_robot(self):
        """Initialize robot connection."""
        if not self.port:
            print("⚠️ No port specified - robot connection will be skipped. Connect USB and restart with --port")
            self.bus = None
            self.all_motors = {}
            return
            
        calib_file = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/so101_chess.json"
        
        with open(calib_file) as f:
            calib_data = json.load(f)
        
        self.all_motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
        
        calibration = {}
        for motor_name, data in calib_data.items():
            calibration[motor_name] = MotorCalibration(
                id=data["id"],
                drive_mode=data["drive_mode"],
                homing_offset=data["homing_offset"],
                range_min=data["range_min"],
                range_max=data["range_max"]
            )
        
        self.bus = FeetechMotorsBus(port=self.port, motors=self.all_motors, calibration=calibration)
    
    def configure_motors_gentle(self):
        """Configure motors 3 (elbow_flex), 4 (wrist_flex), and 6 (gripper) with gentler settings to reduce stiffness.
        
        This reduces PID coefficients and adds torque limits to prevent overload and stiffness.
        """
        if not hasattr(self, 'bus') or not self.bus.is_connected:
            print("⚠️ Cannot configure motors - bus not connected")
            return
        
        print("🔧 Configuring motors 3, 4, and 6 with gentler settings...")
        
        with self.bus.torque_disabled():
            # Configure motor 3 (elbow_flex) - reduce stiffness
            # First verify motor 3 exists and can be read before configuring
            if "elbow_flex" in self.all_motors:
                try:
                    # Verify motor exists by trying to read its position
                    test_read = self.bus.read("Present_Position", "elbow_flex", normalize=False)
                    print(f"  📝 Configuring elbow_flex (motor 3) - current position: {test_read}")
                    
                    self.bus.write("Operating_Mode", "elbow_flex", OperatingMode.POSITION.value)
                    # Reduce P coefficient for less stiffness (default is 32, we use 12)
                    self.bus.write("P_Coefficient", "elbow_flex", 12)
                    # Reduce D coefficient (default is 32, we use 18)
                    self.bus.write("D_Coefficient", "elbow_flex", 18)
                    # Keep I coefficient at 0
                    self.bus.write("I_Coefficient", "elbow_flex", 0)
                    # Add torque limits to prevent overload
                    self.bus.write("Max_Torque_Limit", "elbow_flex", 650)  # 65% of max (slightly higher than wrist)
                    self.bus.write("Protection_Current", "elbow_flex", 325)  # 65% of max current
                    self.bus.write("Overload_Torque", "elbow_flex", 32)  # 32% torque when overloaded
                    print("    ✅ elbow_flex configured with reduced stiffness")
                except Exception as e:
                    print(f"    ⚠️ Could not configure elbow_flex (motor 3): {e}")
                    print(f"    ⏭️ Skipping motor 3 configuration - motor may not be connected or detected")
            
            # Configure motor 4 (wrist_flex) - reduce stiffness
            if "wrist_flex" in self.all_motors:
                try:
                    # Verify motor exists
                    test_read = self.bus.read("Present_Position", "wrist_flex", normalize=False)
                    print(f"  📝 Configuring wrist_flex (motor 4) - current position: {test_read}")
                    
                    self.bus.write("Operating_Mode", "wrist_flex", OperatingMode.POSITION.value)
                    # Reduce P coefficient for less stiffness (default is 32, we use 10)
                    self.bus.write("P_Coefficient", "wrist_flex", 10)
                    # Reduce D coefficient (default is 32, we use 16)
                    self.bus.write("D_Coefficient", "wrist_flex", 16)
                    # Keep I coefficient at 0
                    self.bus.write("I_Coefficient", "wrist_flex", 0)
                    # Add torque limits similar to gripper to prevent overload
                    self.bus.write("Max_Torque_Limit", "wrist_flex", 600)  # 60% of max (more than gripper but still limited)
                    self.bus.write("Protection_Current", "wrist_flex", 300)  # 60% of max current
                    self.bus.write("Overload_Torque", "wrist_flex", 30)  # 30% torque when overloaded
                    print("    ✅ wrist_flex configured with reduced stiffness")
                except Exception as e:
                    print(f"    ⚠️ Could not configure wrist_flex (motor 4): {e}")
            
            # Configure motor 6 (gripper) - further reduce stiffness
            if "gripper" in self.all_motors:
                try:
                    # Verify motor exists
                    test_read = self.bus.read("Present_Position", "gripper", normalize=False)
                    print(f"  📝 Configuring gripper (motor 6) - current position: {test_read}")
                    
                    self.bus.write("Operating_Mode", "gripper", OperatingMode.POSITION.value)
                    # Reduce P coefficient further (default is 32, we use 8)
                    self.bus.write("P_Coefficient", "gripper", 8)
                    # Reduce D coefficient (default is 32, we use 12)
                    self.bus.write("D_Coefficient", "gripper", 12)
                    # Keep I coefficient at 0
                    self.bus.write("I_Coefficient", "gripper", 0)
                    # Reduce torque limits further to prevent overload
                    self.bus.write("Max_Torque_Limit", "gripper", 400)  # 40% of max (reduced from 500)
                    self.bus.write("Protection_Current", "gripper", 200)  # 40% of max current (reduced from 250)
                    self.bus.write("Overload_Torque", "gripper", 20)  # 20% torque when overloaded (reduced from 25)
                    print("    ✅ gripper configured with reduced stiffness")
                except Exception as e:
                    print(f"    ⚠️ Could not configure gripper (motor 6): {e}")
        
        print("✅ Motor configuration complete")
        
    def setup_camera(self):
        """Initialize camera connection for gripper camera only."""
        # Gripper-mounted camera for precision control and vision
        gripper_camera_cfg = OpenCVCameraConfig(
            index_or_path=0,
            width=640,
            height=480,
            fps=30,
            # Use BGR so downstream cv2 conversions (BGR->RGB for Qt/JPEG) are consistent.
            color_mode=ColorMode.BGR,
        )
        self.gripper_camera = OpenCVCamera(gripper_camera_cfg)
        
        # No main camera
        self.main_camera = None
        
        # Store in dict for easier management (following lerobot pattern)
        self.cameras = {
            "gripper": self.gripper_camera
        }
        
        # Maintain backwards compatibility with single camera reference
        self.camera = self.gripper_camera
    
    def setup_kinematics(self):
        """Initialize robot kinematics for forward kinematics calculations."""
        try:
            # Check if URDF exists - try multiple locations
            # Prefer a meshless copy of the provided URDF (preserves exact kinematics without requiring STL assets)
            possible_paths = [
                "./so101_new_calib.nomesh.urdf",  # Preferred: exact kinematics, no mesh files
                "./so101_kinematics.urdf",  # Simplified kinematics-only URDF
                "./so101_new_calib.urdf",  # Full URDF (needs mesh files)
                "./SO101/so101_new_calib.urdf",  # SO101 subdirectory
                Path(__file__).parent / "so101_new_calib.nomesh.urdf",  # Relative to script
                Path(__file__).parent / "so101_kinematics.urdf",  # Relative to script
            ]
            
            urdf_path = None
            for path in possible_paths:
                if Path(path).exists():
                    urdf_path = str(path)
                    print(f"✅ Found URDF at: {urdf_path}")
                    break
            
            if urdf_path is None:
                print("⚠️ URDF not found - using simplified coordinate calculation")
                print(f"   Searched: {[str(p) for p in possible_paths]}")
            
            if urdf_path:
                # IMPORTANT: exclude the jaw joint ("gripper") from IK/FK for end-effector pose.
                # The `gripper_frame_link` is attached to `gripper_link` via a fixed joint, so the jaw DOF does
                # not affect the end-effector pose but can destabilize the IK solver if included.
                joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
                self.kinematics = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name="gripper_frame_link", 
                    joint_names=joint_names
                )
                print("✅ Full kinematics loaded")
            else:
                self.kinematics = None
                print("📐 Using simplified coordinate calculation")
                
        except Exception as e:
            print(f"⚠️ Kinematics initialization failed: {e}")
            self.kinematics = None
    
    def setup_llm(self, api_key: Optional[str] = None, workflow_id: Optional[str] = None):
        """Initialize LLM client using OpenAI ChatKit for natural language control.
        
        ChatKit can be used in two ways:
        1. With workflow_id: Uses OpenAI's hosted Agent Builder workflows via ChatKit sessions
        2. Without workflow_id: Falls back to direct chat.completions API
        
        Note: For full ChatKit server integration, install: pip install openai-chatkit
        """
        self.llm_client = None
        self.llm_enabled = False
        self.chatkit_session = None
        self.workflow_id = workflow_id or os.getenv("OPENAI_CHATKIT_WORKFLOW_ID")
        self.use_chatkit_workflow = bool(self.workflow_id)
        
        if not LLM_AVAILABLE:
            print("⚠️ LLM support not available - install openai package")
            return
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if api_key:
            try:
                self.llm_client = OpenAI(api_key=api_key)
                
                # Try to use ChatKit workflow if workflow_id is provided
                if self.workflow_id:
                    try:
                        # Attempt to create ChatKit session via OpenAI API
                        # Note: This API endpoint may vary - checking for chatkit namespace
                        if hasattr(self.llm_client, 'chatkit') and hasattr(self.llm_client.chatkit, 'sessions'):
                            self.chatkit_session = self.llm_client.chatkit.sessions.create(
                                workflow_id=self.workflow_id
                            )
                            print(f"✅ ChatKit session created with workflow: {self.workflow_id}")
                            self.llm_enabled = True
                        else:
                            # ChatKit API not available in this SDK version
                            print(f"⚠️ ChatKit sessions API not available in this OpenAI SDK version")
                            print(f"   Workflow ID provided: {self.workflow_id}")
                            print("   Falling back to direct chat.completions")
                            print("   Note: For full ChatKit support, you may need:")
                            print("   - A newer version of the openai package, or")
                            print("   - Use the openai-chatkit package for server integration")
                            self.use_chatkit_workflow = False
                            self.llm_enabled = True
                    except Exception as e:
                        print(f"⚠️ Failed to create ChatKit session: {e}")
                        print("   Falling back to direct chat.completions")
                        self.use_chatkit_workflow = False
                        self.llm_enabled = True
                else:
                    # No workflow_id, use direct chat.completions
                    print("ℹ️ Using direct chat.completions API")
                    print("   To use ChatKit workflows, set OPENAI_CHATKIT_WORKFLOW_ID env var or pass --workflow-id")
                    if CHATKIT_SDK_AVAILABLE:
                        print("   ✅ ChatKit Python SDK is available (openai-chatkit)")
                    else:
                        print("   ℹ️  ChatKit Python SDK not installed (optional for server integration)")
                    self.llm_enabled = True
                    
            except Exception as e:
                print(f"⚠️ Failed to initialize LLM client: {e}")
        else:
            print("⚠️ No OpenAI API key provided. Set OPENAI_API_KEY environment variable or use --api-key")
        
    def create_widgets(self):
        """Create all UI widgets."""
        
        # Central widget with proper desktop app spacing
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 15, 20, 15)
        
        # Header section with title and quick stats
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setSpacing(15)
        
        # Main title - terminal style
        title = QLabel("CHESS ROBOT CONTROL")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #58a6ff;")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_layout.addWidget(title, 1)
        
        # Quick status indicators
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        status_layout.setSpacing(5)
        
        # Connection status - terminal style
        self.quick_status = QLabel("[READY]")
        self.quick_status.setStyleSheet("color: #3fb950; font-weight: bold;")
        self.quick_status.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.quick_status)
        
        # Update rate indicator
        update_label = QLabel("10 Hz")
        update_label.setStyleSheet("color: #8b949e;")
        update_label.setAlignment(Qt.AlignCenter)
        update_label.setToolTip("Update rate: 10 updates per second")
        status_layout.addWidget(update_label)
        
        header_layout.addWidget(status_widget)
        main_layout.addWidget(header_widget)
        
        # Create main horizontal layout for panels with solid desktop spacing
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(15)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        
        # LEFT COLUMN: Camera and Motors
        left_column = QVBoxLayout()
        left_column.setSpacing(15)
        self.create_camera_panel(left_column)
        self.create_motor_panel(left_column)
        panels_layout.addLayout(left_column, 1)  # Left column gets 1 part
        
        # RIGHT COLUMN: Full ChatKit/LLM Control
        right_column = QVBoxLayout()
        right_column.setSpacing(15)
        self.create_llm_panel(right_column)
        panels_layout.addLayout(right_column, 1)  # Right column gets 1 part (equal split)
        
        main_layout.addLayout(panels_layout, 1)
        
        # Bottom: Control buttons
        self.create_control_panel(main_layout)
        
    def create_camera_panel(self, parent_layout):
        """Create camera view panel (gripper camera only)."""
        # Container for camera
        cameras_container = QWidget()
        cameras_layout = QVBoxLayout(cameras_container)
        cameras_layout.setSpacing(8)
        cameras_layout.setContentsMargins(0, 0, 0, 0)
        
        # Gripper camera panel - now the primary camera
        gripper_camera_group = QGroupBox("Camera [Gripper View]")
        gripper_camera_group.setToolTip("Gripper-mounted camera - for vision and precision control")
        gripper_camera_layout = QVBoxLayout(gripper_camera_group)
        gripper_camera_layout.setSpacing(6)
        gripper_camera_layout.setContentsMargins(8, 8, 8, 8)
        
        self.gripper_camera_label = QLabel("Initializing camera...")
        self.gripper_camera_label.setAlignment(Qt.AlignCenter)
        self.gripper_camera_label.setMinimumSize(480, 360)
        self.gripper_camera_label.setMaximumSize(640, 480)
        self.gripper_camera_label.setScaledContents(True)
        gripper_camera_layout.addWidget(self.gripper_camera_label)
        
        self.gripper_camera_status = QLabel("[CONNECTING]")
        self.gripper_camera_status.setStyleSheet("color: #f0883e;")
        self.gripper_camera_status.setAlignment(Qt.AlignCenter)
        gripper_camera_layout.addWidget(self.gripper_camera_status)
        
        cameras_layout.addWidget(gripper_camera_group)
        
        # Dummy main camera labels (for compatibility - not displayed)
        self.main_camera_label = QLabel()
        self.main_camera_status = QLabel()
        
        # FPS counter
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #7d8590; font-size: 9pt;")
        self.fps_label.setAlignment(Qt.AlignCenter)
        cameras_layout.addWidget(self.fps_label)
        
        parent_layout.addWidget(cameras_container)
        
    def create_motor_panel(self, parent_layout):
        """Create motor status panel."""
        motor_group = QGroupBox("Motors")
        # Terminal styling applied globally
        motor_group.setToolTip("Real-time motor signals: position, velocity, load, voltage, temperature, current, and control signals.")
        motor_layout = QVBoxLayout(motor_group)
        motor_layout.setSpacing(12)
        motor_layout.setContentsMargins(12, 12, 12, 12)
        
        # Motor signals storage - comprehensive data structure
        self.motor_labels = {}
        self.motor_signal_labels = {}  # Store all signal labels per motor
        self.motor_status_labels = {}  # Store status indicators per motor
        
        # Create scrollable area for motor details
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        
        # Create detailed motor displays
        for motor_name in self.all_motors.keys():
            motor_frame = QFrame()
            motor_frame_layout = QVBoxLayout(motor_frame)
            motor_frame_layout.setSpacing(6)
            
            # Motor header with name and status
            header_layout = QHBoxLayout()
            
            motor_name_label = QLabel(f"> {motor_name.replace('_', ' ').upper()}")
            motor_name_label.setStyleSheet("color: #58a6ff; font-weight: bold;")
            header_layout.addWidget(motor_name_label)
            
            header_layout.addStretch()
            
            # Status indicator
            status_label = QLabel("[ ]")
            status_label.setStyleSheet("color: #6e7681;")
            status_label.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(status_label)
            
            motor_frame_layout.addLayout(header_layout)
            
            # Signals grid - comprehensive display
            signals_grid = QGridLayout()
            signals_grid.setSpacing(6)
            signals_grid.setColumnStretch(0, 1)  # Signal name
            signals_grid.setColumnStretch(1, 1)  # Signal value
            
            # Define all available signals
            signal_configs = [
                ("Position", "position", "°", "#3498db"),
                ("Goal Position", "goal_position", "°", "#2980b9"),
                ("Velocity", "velocity", "steps/s", "#27ae60"),
                ("Load", "load", "%", "#e67e22"),
                ("Current", "current", "mA", "#e74c3c"),
                ("Voltage", "voltage", "V", "#f39c12"),
                ("Temperature", "temperature", "°C", "#9b59b6"),
                ("Moving", "moving", "", "#95a5a6"),
                ("Torque", "torque_enable", "", "#16a085"),
            ]
            
            signal_labels = {}
            for idx, (signal_name, signal_key, unit, color) in enumerate(signal_configs):
                # Signal name - terminal style
                name_lbl = QLabel(f"{signal_name}:")
                name_lbl.setStyleSheet("color: #8b949e;")
                signals_grid.addWidget(name_lbl, idx, 0)
                
                # Signal value - terminal style
                value_lbl = QLabel("--")
                value_lbl.setStyleSheet("color: #c9d1d9;")
                value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                signals_grid.addWidget(value_lbl, idx, 1)
                
                signal_labels[signal_key] = value_lbl
            
            motor_frame_layout.addLayout(signals_grid)
            scroll_layout.addWidget(motor_frame)
            
            # Store labels for updates
            self.motor_labels[motor_name] = signal_labels.get("position")
            self.motor_status_labels[motor_name] = status_label
            self.motor_signal_labels[motor_name] = signal_labels
        
        scroll.setWidget(scroll_widget)
        motor_layout.addWidget(scroll)
        
        # Overall robot status - terminal style
        self.robot_status = QLabel("[CONNECTING]")
        self.robot_status.setStyleSheet("color: #f0883e; font-weight: bold;")
        self.robot_status.setAlignment(Qt.AlignCenter)
        motor_layout.addWidget(self.robot_status)
        
        parent_layout.addWidget(motor_group)
        
    def create_chess_panel(self, parent_layout):
        """Create chess board visualization panel."""
        chess_group = QGroupBox("♟️ Chess Board")
        chess_group.setStyleSheet("""
            QGroupBox {
                font-size: 11pt;
                font-weight: 600;
                color: #ecf0f1;
                background: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 8px;
                padding-top: 22px;
                padding-bottom: 18px;
                padding-left: 18px;
                padding-right: 18px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 18px;
                padding: 0 10px 0 10px;
                color: #27ae60;
            }
        """)
        chess_group.setToolTip("Chess board visualization. Shows which square the robot is currently over.")
        chess_layout = QVBoxLayout(chess_group)
        chess_layout.setSpacing(12)
        chess_layout.setContentsMargins(12, 12, 12, 12)
        
        # Chess board widget
        self.chess_board = ChessBoardWidget()
        chess_layout.addWidget(self.chess_board, alignment=Qt.AlignCenter)
        
        # Robot position details - solid desktop styling
        self.robot_details = QLabel("Position: -- | Accuracy: --")
        self.robot_details.setStyleSheet("""
            font-size: 9pt;
            font-weight: 500;
            color: #bdc3c7;
            padding: 10px 0px;
        """)
        self.robot_details.setAlignment(Qt.AlignCenter)
        chess_layout.addWidget(self.robot_details)
        
        # Robot position indicator - solid desktop styling
        self.robot_position_label = QLabel("📍 --")
        self.robot_position_label.setStyleSheet("""
            font-size: 11pt;
            font-weight: 600;
            color: #e74c3c;
            padding: 10px 18px;
            background-color: #3d1e1e;
            border-radius: 6px;
            border: 1px solid #e74c3c;
        """)
        self.robot_position_label.setAlignment(Qt.AlignCenter)
        chess_layout.addWidget(self.robot_position_label)
        
        parent_layout.addWidget(chess_group)
    
    def create_coordinates_panel(self, parent_layout):
        """Create robot base coordinates panel."""
        coord_group = QGroupBox("🔧 Coordinates")
        coord_group.setStyleSheet("""
            QGroupBox {
                font-size: 11pt;
                font-weight: 600;
                color: #ecf0f1;
                background: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 8px;
                padding-top: 22px;
                padding-bottom: 18px;
                padding-left: 18px;
                padding-right: 18px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 18px;
                padding: 0 10px 0 10px;
                color: #f39c12;
            }
        """)
        coord_group.setToolTip("End-effector position in robot base frame. X, Y, Z coordinates in millimeters.")
        coord_layout = QVBoxLayout(coord_group)
        coord_layout.setSpacing(10)
        coord_layout.setContentsMargins(12, 12, 12, 12)
        
        # End-effector position in base frame - solid desktop styling
        self.ee_position_label = QLabel("End-Effector Position:")
        self.ee_position_label.setStyleSheet("""
            font-size: 10pt;
            font-weight: 600;
            color: #ecf0f1;
            padding: 8px 0px;
        """)
        coord_layout.addWidget(self.ee_position_label)
        
        # X, Y, Z coordinates - solid desktop display
        coord_grid = QGridLayout()
        coord_grid.setSpacing(8)
        
        self.coord_x = QLabel("X: ---.-- mm")
        self.coord_x.setStyleSheet("""
            font-size: 11pt;
            font-weight: 600;
            color: #e74c3c;
            padding: 10px 14px;
            background-color: #1e2329;
            border-radius: 4px;
            border: 1px solid #34495e;
        """)
        self.coord_x.setAlignment(Qt.AlignCenter)
        coord_grid.addWidget(self.coord_x, 0, 0)
        
        self.coord_y = QLabel("Y: ---.-- mm")
        self.coord_y.setStyleSheet("""
            font-size: 11pt;
            font-weight: 600;
            color: #27ae60;
            padding: 10px 14px;
            background-color: #1e2329;
            border-radius: 4px;
            border: 1px solid #34495e;
        """)
        self.coord_y.setAlignment(Qt.AlignCenter)
        coord_grid.addWidget(self.coord_y, 0, 1)
        
        self.coord_z = QLabel("Z: ---.-- mm")
        self.coord_z.setStyleSheet("""
            font-size: 11pt;
            font-weight: 600;
            color: #3498db;
            padding: 10px 14px;
            background-color: #1e2329;
            border-radius: 4px;
            border: 1px solid #34495e;
        """)
        self.coord_z.setAlignment(Qt.AlignCenter)
        coord_grid.addWidget(self.coord_z, 0, 2)
        
        coord_layout.addLayout(coord_grid)
        
        # Workspace info - solid desktop styling
        workspace_label = QLabel("Workspace Status:")
        workspace_label.setStyleSheet("""
            font-size: 10pt;
            font-weight: 600;
            color: #ecf0f1;
            padding: 10px 0px 5px 0px;
        """)
        coord_layout.addWidget(workspace_label)
        
        self.workspace_status = QLabel("Position: -- | Reach: --")
        self.workspace_status.setStyleSheet("""
            font-size: 9pt;
            font-weight: 500;
            color: #bdc3c7;
            padding: 6px 0px;
        """)
        coord_layout.addWidget(self.workspace_status)
        
        # Distance from base - solid desktop styling
        self.distance_label = QLabel("Distance: --- mm")
        self.distance_label.setStyleSheet("""
            font-size: 10pt;
            font-weight: 600;
            color: #f39c12;
            padding: 10px 14px;
            background-color: #3d2817;
            border-radius: 4px;
            border: 1px solid #f39c12;
        """)
        self.distance_label.setAlignment(Qt.AlignCenter)
        coord_layout.addWidget(self.distance_label)
        
        # Joint configuration display - solid desktop formatting
        joint_label = QLabel("Joint Configuration:")
        joint_label.setStyleSheet("""
            font-size: 10pt;
            font-weight: 600;
            color: #ecf0f1;
            padding: 10px 0px 5px 0px;
        """)
        coord_layout.addWidget(joint_label)
        
        self.joint_config = QLabel("Shoulder: --°, --° | Elbow: --° | Wrist: --°, --°")
        self.joint_config.setStyleSheet("""
            font-size: 8pt;
            font-weight: 500;
            color: #95a5a6;
            padding: 6px 0px;
        """)
        self.joint_config.setWordWrap(True)
        coord_layout.addWidget(self.joint_config)
        
        # Simple 2D workspace visualization
        top_view_label = QLabel("Top View:")
        top_view_label.setStyleSheet("""
            font-size: 10pt;
            font-weight: 600;
            color: #ecf0f1;
            padding: 10px 0px 5px 0px;
        """)
        coord_layout.addWidget(top_view_label)
        
        self.workspace_canvas = WorkspaceWidget(view_type='top')
        coord_layout.addWidget(self.workspace_canvas, alignment=Qt.AlignCenter)
        
        # Draw workspace
        self.workspace_canvas.update()
        
        parent_layout.addWidget(coord_group)
    
    def create_robot_3d_panel(self, parent_layout):
        """Create 3D robot visualization panel."""
        robot_3d_group = QGroupBox("3D View")
        robot_3d_layout = QVBoxLayout(robot_3d_group)
        robot_3d_layout.setSpacing(8)
        robot_3d_layout.setContentsMargins(8, 8, 8, 8)
        
        # Create 3D robot widget
        self.robot_3d_widget = Robot3DWidget()
        self.robot_3d_widget.setMinimumSize(400, 350)
        self.robot_3d_widget.setMaximumSize(800, 600)
        robot_3d_layout.addWidget(self.robot_3d_widget, alignment=Qt.AlignCenter)
        
        # Info label - terminal style
        info_label = QLabel("Drag to rotate | Scroll to zoom")
        info_label.setStyleSheet("color: #8b949e;")
        info_label.setAlignment(Qt.AlignCenter)
        robot_3d_layout.addWidget(info_label)
        
        parent_layout.addWidget(robot_3d_group, 1)
        
    def create_control_panel(self, parent_layout):
        """Create control buttons panel."""
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # Control buttons - terminal style
        go_home_btn = QPushButton("HOME")
        go_home_btn.setToolTip("Move robot to home position (Ctrl+H)")
        go_home_btn.clicked.connect(self.go_home)
        QShortcut(QKeySequence("Ctrl+H"), self).activated.connect(self.go_home)
        control_layout.addWidget(go_home_btn)
        
        refresh_btn = QPushButton("REFRESH")
        refresh_btn.setToolTip("Refresh all displays (F5)")
        refresh_btn.clicked.connect(self.refresh_status)
        QShortcut(QKeySequence("F5"), self).activated.connect(self.refresh_status)
        control_layout.addWidget(refresh_btn)
        
        # Restart button - useful for dev mode
        restart_btn = QPushButton("RESTART")
        restart_btn.setToolTip("Restart the application (Ctrl+R)")
        restart_btn.clicked.connect(self.restart_app)
        restart_btn.setStyleSheet("""
            QPushButton {
                background: #1f6feb;
                color: white;
            }
            QPushButton:hover {
                background: #388bfd;
            }
        """)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.restart_app)
        control_layout.addWidget(restart_btn)
        
        # Calibrate button - for chess board calibration
        calibrate_btn = QPushButton("CALIBRATE BOARD")
        calibrate_btn.setToolTip("Calibrate chess board positions (Ctrl+B)")
        calibrate_btn.clicked.connect(self.start_calibration)
        calibrate_btn.setStyleSheet("""
            QPushButton {
                background: #a371f7;
                color: white;
            }
            QPushButton:hover {
                background: #8957e5;
            }
        """)
        QShortcut(QKeySequence("Ctrl+B"), self).activated.connect(self.start_calibration)
        control_layout.addWidget(calibrate_btn)
        
        # Test Motors button - move each motor a bit
        test_motors_btn = QPushButton("TEST MOTORS")
        test_motors_btn.setToolTip("Test each motor individually with small movements (Ctrl+T)")
        test_motors_btn.clicked.connect(self.test_motors)
        test_motors_btn.setStyleSheet("""
            QPushButton {
                background: #f0883e;
                color: white;
            }
            QPushButton:hover {
                background: #ff9a4d;
            }
        """)
        QShortcut(QKeySequence("Ctrl+T"), self).activated.connect(self.test_motors)
        control_layout.addWidget(test_motors_btn)
        
        # Manual Motor Control button
        manual_control_btn = QPushButton("MANUAL CONTROL")
        manual_control_btn.setToolTip("Open manual motor control panel (Ctrl+M)")
        manual_control_btn.clicked.connect(self.open_manual_control)
        manual_control_btn.setStyleSheet("""
            QPushButton {
                background: #8957e5;
                color: white;
            }
            QPushButton:hover {
                background: #a371f7;
            }
        """)
        QShortcut(QKeySequence("Ctrl+M"), self).activated.connect(self.open_manual_control)
        control_layout.addWidget(manual_control_btn)
        
        # Position Calibration button
        positions_btn = QPushButton("POSITIONS")
        positions_btn.setToolTip("Save and recall arm positions (Ctrl+P)")
        positions_btn.clicked.connect(self.open_position_calibration)
        positions_btn.setStyleSheet("""
            QPushButton {
                background: #238636;
                color: white;
            }
            QPushButton:hover {
                background: #2ea043;
            }
        """)
        QShortcut(QKeySequence("Ctrl+P"), self).activated.connect(self.open_position_calibration)
        control_layout.addWidget(positions_btn)

        # Workspace estimate button
        workspace_btn = QPushButton("WORKSPACE")
        workspace_btn.setToolTip("Estimate reachable workspace from calibration limits (no motion) (Ctrl+W)")
        workspace_btn.clicked.connect(self.run_workspace_estimate)
        QShortcut(QKeySequence("Ctrl+W"), self).activated.connect(self.run_workspace_estimate)
        control_layout.addWidget(workspace_btn)

        # Board view localization (motor+camera) - find shoulder_pan center for the board
        localize_btn = QPushButton("LOCALIZE")
        localize_btn.setToolTip("Calibrate board-in-view centering (uses camera + shoulder_pan) (Ctrl+L)")
        localize_btn.clicked.connect(self.run_board_localize)
        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(self.run_board_localize)
        control_layout.addWidget(localize_btn)

        # Pick Pawn button (closed-loop, gripper-camera-only)
        pick_pawn_btn = QPushButton("PICK PAWN")
        pick_pawn_btn.setToolTip("Find the pawn and pick it up using only the gripper camera (Ctrl+G)")
        pick_pawn_btn.clicked.connect(self.pick_pawn)
        pick_pawn_btn.setStyleSheet("""
            QPushButton {
                background: #2ea043;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #3fb950;
            }
        """)
        QShortcut(QKeySequence("Ctrl+G"), self).activated.connect(self.pick_pawn)
        control_layout.addWidget(pick_pawn_btn)
        
        control_layout.addStretch()
        
        stop_btn = QPushButton("STOP")
        stop_btn.setToolTip("Stop monitoring and close application (Ctrl+Q)")
        stop_btn.clicked.connect(self.stop_monitoring)
        QShortcut(QKeySequence("Ctrl+Q"), self).activated.connect(self.stop_monitoring)
        control_layout.addWidget(stop_btn)
        
        # Status bar - terminal style
        self.status_bar = QLabel("> Initializing...")
        self.status_bar.setStyleSheet("color: #8b949e;")
        self.status_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        control_layout.addWidget(self.status_bar, 1)
        
        parent_layout.addWidget(control_frame)

    def pick_pawn(self):
        """Convenience UI entrypoint for the auto pick routine."""
        if not self.llm_enabled or not self.llm_client:
            self.status_bar.setText("❌ LLM not available (needed for pawn localization)")
            return
        # Reuse existing threaded execution path (so UI stays responsive)
        self.llm_command_input.setPlainText("pick pawn")
        self._start_llm_execution_thread()

    def run_workspace_estimate(self):
        """Recompute workspace estimate (no motion) and display summary in logs."""
        try:
            self._append_exec_log("info", "Workspace: estimating reachable XYZ bounds (no motion)...")
            est = self._estimate_workspace_from_calibration(sample_count=1800)
            self.workspace_estimate = est
            self._save_workspace_estimate(est)
            b = (est or {}).get("ee_percentiles_mm") or {}
            jl = (est or {}).get("joint_limits_deg") or {}
            msg = (
                f"Workspace estimate saved. EE p5/p95 (mm): "
                f"x={b.get('x_p5', 0):.0f}..{b.get('x_p95', 0):.0f}, "
                f"y={b.get('y_p5', 0):.0f}..{b.get('y_p95', 0):.0f}, "
                f"z={b.get('z_p5', 0):.0f}..{b.get('z_p95', 0):.0f}"
            )
            self._append_exec_log("info", msg)
            # Also show which way the camera should look (recommended scan pose)
            rec = (est or {}).get("recommended_scan_pose_deg") or {}
            if rec:
                self._append_exec_log("info", f"Recommended scan pose (deg): {json.dumps(rec)}")
        except Exception as e:
            self._append_exec_log("error", f"Workspace estimate failed: {e}")

    def run_board_localize(self):
        """Vision+motor localization: learn what shoulder_pan angle centers the board in the camera view.

        This is a safe routine: it only sweeps shoulder_pan (motor 1) in small increments and uses the LLM
        (vision) to detect the chessboard center/bbox in the camera image (no OpenCV detection).
        """
        if not getattr(self, "bus", None) or not getattr(self.bus, "is_connected", False):
            self._append_exec_log("error", "Localize: robot bus not connected")
            return
        if not getattr(self, "gripper_camera", None) or not getattr(self.gripper_camera, "is_connected", False):
            self._append_exec_log("error", "Localize: gripper camera not connected")
            return
        if not getattr(self, "llm_enabled", False) or not getattr(self, "llm_client", None):
            self._append_exec_log("error", "Localize: LLM not available (needed for vision-based board detection)")
            return
        if self.kinematics is None:
            self._append_exec_log("warning", "Localize: kinematics not available (continuing anyway)")

        # Pause monitoring thread to avoid serial conflicts during reads/writes
        was_paused = False
        if hasattr(self, "monitor_thread") and self.monitor_thread:
            was_paused = self.monitor_thread.paused
            self.monitor_thread.paused = True
            time.sleep(0.2)

        try:
            self._append_exec_log("info", "Localize: starting board centering scan (pan-only)...")

            # Read current joint positions (best effort)
            self.update_position_model()
            cur = dict(self.position_model.get("joints", {}))
            pan = float(cur.get("shoulder_pan", 0.0))
            roll = float(cur.get("wrist_roll", 0.0))

            # Pan limits from workspace estimate if available
            pan_lo, pan_hi = -90.0, 90.0
            try:
                jl = (getattr(self, "workspace_estimate", None) or {}).get("joint_limits_deg") or {}
                if isinstance(jl.get("shoulder_pan"), dict):
                    pan_lo = float(jl["shoulder_pan"].get("min", pan_lo))
                    pan_hi = float(jl["shoulder_pan"].get("max", pan_hi))
            except Exception:
                pass

            # Sweep targets (15° increments, bounded)
            span = 75.0
            step = 15.0
            raw_targets = [pan + d for d in np.arange(-span, span + 0.1, step)]
            targets = []
            for t in raw_targets:
                tt = float(np.clip(t, pan_lo, pan_hi))
                if not targets or abs(tt - targets[-1]) > 1e-3:
                    targets.append(tt)

            best = None
            best_pan = None
            best_errx = None
            best_area = None

            for tpan in targets:
                # Move pan only (small deltas between steps)
                act = {"shoulder_pan.pos": float(tpan)}
                safe = self._validate_llm_action(act, cur)
                if safe:
                    self._execute_llm_action(safe)
                    time.sleep(0.5)
                    self.update_position_model()
                    cur = dict(self.position_model.get("joints", {}))

                # LLM vision detection (board bbox/center)
                img = self.capture_camera_image("gripper")
                det = self._llm_locate_chessboard(img) if img else None
                if det and det.get("found") and float(det.get("confidence", 0.0)) >= 0.25:
                    cx = float((det.get("center_norm") or {}).get("x", 0.5))
                    errx = cx - 0.5
                    bb = det.get("bbox_norm") or {}
                    area = max(0.0, float(bb.get("x2", 0.0)) - float(bb.get("x1", 0.0))) * max(
                        0.0, float(bb.get("y2", 0.0)) - float(bb.get("y1", 0.0))
                    )
                    conf = float(det.get("confidence", 0.0))
                    ang = det.get("angle_deg", None)
                    msg = f"Localize: pan={tpan:+.1f}° board_x={cx:.3f} errx={errx:+.3f} conf={conf:.2f} area={area:.3f} angle={ang}"
                    self._append_exec_log("info", msg)

                    # Choose by smallest |errx|; tie-break by larger area*confidence
                    tie_score = area * conf
                    best_tie = (best_area or 0.0) * float((best or {}).get("confidence", 0.0)) if best else -1.0
                    if (
                        best is None
                        or abs(errx) < abs(best_errx)
                        or (abs(errx) == abs(best_errx) and tie_score > best_tie)
                    ):
                        best = det
                        best_pan = float(tpan)
                        best_errx = float(errx)
                        best_area = float(area)
                else:
                    conf = float((det or {}).get("confidence", 0.0)) if det else 0.0
                    noted = (det or {}).get("notes", "") if det else ""
                    self._append_exec_log("warning", f"Localize: pan={tpan:+.1f}° board not detected (conf={conf:.2f}) {noted}".strip())

            if not best:
                self._append_exec_log("error", "Localize: chessboard not detected anywhere in sweep. Try an OVERVIEW pose and rerun.")
                return

            # Refine pan center with a small probe (optional)
            probe_step = 5.0
            pan_center = best_pan
            try:
                # At best_pan, estimate sensitivity d(errx)/d(pan)
                det0 = best
                err0 = float((det0.get("center_norm") or {}).get("x", 0.5)) - 0.5
                pan2 = float(np.clip(best_pan + probe_step, pan_lo, pan_hi))
                if abs(pan2 - best_pan) > 1e-3:
                    safe = self._validate_llm_action({"shoulder_pan.pos": pan2}, cur)
                    if safe:
                        self._execute_llm_action(safe)
                        time.sleep(0.5)
                        img2 = self.capture_camera_image("gripper")
                        det1 = self._llm_locate_chessboard(img2) if img2 else None
                        # Go back
                        safe = self._validate_llm_action({"shoulder_pan.pos": best_pan}, cur)
                        if safe:
                            self._execute_llm_action(safe)
                            time.sleep(0.4)
                        if det1 and det1.get("found") and float(det1.get("confidence", 0.0)) >= 0.25:
                            err1 = float((det1.get("center_norm") or {}).get("x", 0.5)) - 0.5
                            sens = (err1 - err0) / (pan2 - best_pan)
                            if abs(sens) > 1e-4:
                                delta = float(np.clip(-err0 / sens, -12.0, 12.0))
                                pan_center = float(np.clip(best_pan + delta, pan_lo, pan_hi))
            except Exception:
                pass

            # Save calibration
            calib = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "pan_center_deg": float(pan_center),
                "pan_limits_deg": {"min": float(pan_lo), "max": float(pan_hi)},
                "wrist_roll_at_calibration_deg": float(roll),
                "board_center_norm": best.get("center_norm"),
                "board_angle_deg": best.get("angle_deg", None),
                "note": "pan_center_deg is the shoulder_pan angle that best centers the detected board in the image.",
            }
            self.board_view_calibration = calib
            self._save_board_view_calibration(calib)

            # Report key metric the user asked for: m1 (shoulder_pan) centeredness
            self.update_position_model()
            cur = dict(self.position_model.get("joints", {}))
            cur_pan = float(cur.get("shoulder_pan", 0.0))
            pan_err = cur_pan - float(pan_center)
            self._append_exec_log("info", f"✅ Localize saved: pan_center={pan_center:+.1f}°. Current m1 error={pan_err:+.1f}° (shoulder_pan - pan_center).")
            self._append_exec_log("info", f"Tip: keep wrist_roll near 0° while scanning; current roll={float(cur.get('wrist_roll', 0.0)):+.1f}°")
        finally:
            if hasattr(self, "monitor_thread") and self.monitor_thread:
                self.monitor_thread.paused = was_paused
        
    def create_workspace_control_panel(self, parent_layout):
        """Create workspace visualization and control panel."""
        workspace_group = QGroupBox("🎯 Workspace Control")
        workspace_group.setStyleSheet("""
            QGroupBox {
                font-size: 11pt;
                font-weight: 600;
                color: #ecf0f1;
                background: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 8px;
                padding-top: 22px;
                padding-bottom: 18px;
                padding-left: 18px;
                padding-right: 18px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 18px;
                padding: 0 10px 0 10px;
                color: #3498db;
            }
        """)
        workspace_group.setToolTip("Workspace visualization and interactive gripper control. Move robot end-effector with buttons.")
        workspace_layout = QHBoxLayout(workspace_group)
        workspace_layout.setSpacing(18)
        workspace_layout.setContentsMargins(12, 12, 12, 12)
        
        # Visualization section
        viz_layout = QVBoxLayout()
        viz_layout.setSpacing(10)
        
        side_view_label = QLabel("Workspace Volume (Side View)")
        side_view_label.setStyleSheet("""
            font-size: 10pt;
            font-weight: bold;
            color: #ecf0f1;
            padding: 5px 0px;
        """)
        side_view_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(side_view_label)
        
        self.workspace_side_canvas = WorkspaceWidget(view_type='side')
        viz_layout.addWidget(self.workspace_side_canvas, alignment=Qt.AlignCenter)
        
        top_view_label = QLabel("Workspace Volume (Top View)")
        top_view_label.setStyleSheet("""
            font-size: 10pt;
            font-weight: bold;
            color: #ecf0f1;
            padding: 5px 0px;
        """)
        top_view_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(top_view_label)
        
        self.workspace_top_canvas = WorkspaceWidget(view_type='top')
        self.workspace_top_canvas.setFixedSize(300, 200)
        viz_layout.addWidget(self.workspace_top_canvas, alignment=Qt.AlignCenter)
        
        workspace_layout.addLayout(viz_layout)
        
        # Control section
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)
        
        control_title = QLabel("Interactive Gripper Control")
        control_title.setStyleSheet("""
            font-size: 12pt;
            font-weight: bold;
            color: #ecf0f1;
            padding: 5px 0px;
        """)
        control_title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(control_title)
        
        # Current position display - better formatting
        self.current_pos_label = QLabel("Current Position:\nX: --- mm | Y: --- mm | Z: --- mm")
        self.current_pos_label.setStyleSheet("""
            font-size: 10pt;
            color: #bdc3c7;
            padding: 5px;
        """)
        control_layout.addWidget(self.current_pos_label)
        
        # Movement controls
        move_group = QGroupBox("Move Gripper")
        move_group.setStyleSheet("""
            QGroupBox {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #95a5a6;
                border-radius: 3px;
                padding-top: 10px;
            }
        """)
        move_layout = QVBoxLayout(move_group)
        
        # X-axis controls
        x_frame = QWidget()
        x_layout = QHBoxLayout(x_frame)
        x_label = QLabel("X:")
        x_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #e74c3c; min-width: 30px;")
        x_layout.addWidget(x_label)
        
        back_btn = QPushButton("←Back")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 9pt;
                font-weight: bold;
                padding: 8px 12px;
                border: 1px solid #c0392b;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        back_btn.setToolTip("Move gripper backward (negative X direction)")
        back_btn.clicked.connect(lambda: self.move_gripper(-10, 0, 0))
        x_layout.addWidget(back_btn)
        
        fwd_btn = QPushButton("Fwd→")
        fwd_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 9pt;
                font-weight: bold;
                padding: 8px 12px;
                border: 1px solid #c0392b;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        fwd_btn.setToolTip("Move gripper forward (positive X direction)")
        fwd_btn.clicked.connect(lambda: self.move_gripper(10, 0, 0))
        x_layout.addWidget(fwd_btn)
        move_layout.addWidget(x_frame)
        
        # Y-axis controls
        y_frame = QWidget()
        y_layout = QHBoxLayout(y_frame)
        y_label = QLabel("Y:")
        y_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #27ae60; min-width: 30px;")
        y_layout.addWidget(y_label)
        
        left_btn = QPushButton("←Left")
        left_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 9pt;
                font-weight: bold;
                padding: 8px 12px;
                border: 1px solid #229954;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        left_btn.setToolTip("Move gripper left (negative Y direction)")
        left_btn.clicked.connect(lambda: self.move_gripper(0, -10, 0))
        y_layout.addWidget(left_btn)
        
        right_btn = QPushButton("Right→")
        right_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 9pt;
                font-weight: bold;
                padding: 8px 12px;
                border: 1px solid #229954;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        right_btn.setToolTip("Move gripper right (positive Y direction)")
        right_btn.clicked.connect(lambda: self.move_gripper(0, 10, 0))
        y_layout.addWidget(right_btn)
        move_layout.addWidget(y_frame)
        
        # Z-axis controls
        z_frame = QWidget()
        z_layout = QHBoxLayout(z_frame)
        z_label = QLabel("Z:")
        z_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #3498db; min-width: 30px;")
        z_layout.addWidget(z_label)
        
        down_btn = QPushButton("↓Down")
        down_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 9pt;
                font-weight: bold;
                padding: 8px 12px;
                border: 1px solid #2980b9;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        down_btn.setToolTip("Move gripper down (negative Z direction)")
        down_btn.clicked.connect(lambda: self.move_gripper(0, 0, -10))
        z_layout.addWidget(down_btn)
        
        up_btn = QPushButton("Up↑")
        up_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 9pt;
                font-weight: bold;
                padding: 8px 12px;
                border: 1px solid #2980b9;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        up_btn.setToolTip("Move gripper up (positive Z direction)")
        up_btn.clicked.connect(lambda: self.move_gripper(0, 0, 10))
        z_layout.addWidget(up_btn)
        move_layout.addWidget(z_frame)
        
        # Step size control
        size_frame = QWidget()
        size_layout = QHBoxLayout(size_frame)
        size_label = QLabel("Step Size:")
        size_label.setStyleSheet("font-size: 9pt; color: #ecf0f1;")
        size_layout.addWidget(size_label)
        
        self.step_size_combo = QComboBox()
        self.step_size_combo.addItems(["5", "10", "20", "50"])
        self.step_size_combo.setCurrentText("10")
        self.step_size_combo.setStyleSheet("""
            QComboBox {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 2px solid #3498db;
                border-radius: 4px;
                padding: 5px;
                min-width: 70px;
                font-size: 9pt;
            }
            QComboBox:hover {
                border-color: #2980b9;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border: 2px solid #3498db;
                width: 8px;
                height: 8px;
            }
        """)
        self.step_size_combo.setToolTip("Step size for gripper movement (in millimeters)")
        size_layout.addWidget(self.step_size_combo)
        move_layout.addWidget(size_frame)
        
        control_layout.addWidget(move_group)
        
        # Workspace bounds display
        bounds_group = QGroupBox("Workspace Bounds")
        bounds_group.setStyleSheet("""
            QGroupBox {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #95a5a6;
                border-radius: 3px;
                padding-top: 10px;
            }
        """)
        bounds_layout = QVBoxLayout(bounds_group)
        
        self.workspace_bounds = QLabel("X: [---,---] mm\nY: [---,---] mm\nZ: [---,---] mm")
        self.workspace_bounds.setStyleSheet("""
            font-size: 9pt;
            color: #95a5a6;
            padding: 5px;
        """)
        bounds_layout.addWidget(self.workspace_bounds)
        
        control_layout.addWidget(bounds_group)
        control_layout.addStretch()
        
        workspace_layout.addLayout(control_layout)
        
        # Draw initial workspace
        self.draw_workspace_volume()
        
        parent_layout.addWidget(workspace_group)
    
    def draw_workspace_volume(self):
        """Draw the robot's reachable workspace volume."""
        # The workspace is drawn in the custom widget's paintEvent
        self.workspace_side_canvas.update()
        self.workspace_top_canvas.update()
    
    def create_llm_panel(self, parent_layout):
        """Create simple, functional LLM control panel."""
        llm_group = QGroupBox("LLM Control")
        llm_layout = QVBoxLayout(llm_group)
        llm_layout.setSpacing(10)
        llm_layout.setContentsMargins(12, 15, 12, 12)

        # Keep a reference so we can adjust prominence during execution
        self._llm_panel_layout = llm_layout
        
        # Model selection row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.addItems([
            # Latest GPT-5 models (vision-capable)
            "gpt-5.1 👁️", "gpt-5-mini 👁️", "gpt-5-nano 👁️",
            # GPT-4.1 models (vision-capable)  
            "gpt-4.1 👁️", "gpt-4.1-mini 👁️", "gpt-4.1-nano 👁️",
            # GPT-4o models (vision-capable)
            "gpt-4o 👁️", "gpt-4o-mini 👁️",
            # GPT-4 vision models
            "gpt-4-turbo 👁️", "gpt-4-vision-preview 👁️",
            # o-series reasoning models (text-only)
            "o3", "o4-mini", "o1-preview", "o1-mini",
            # Legacy (text-only)
            "gpt-4", "gpt-3.5-turbo"
        ])
        self.llm_model_combo.setMinimumWidth(150)
        self.llm_model_combo.currentTextChanged.connect(self.on_model_changed)
        model_row.addWidget(self.llm_model_combo)
        
        # Reasoning effort selector (for o-series models)
        model_row.addWidget(QLabel("Reasoning:"))
        self.reasoning_effort_combo = QComboBox()
        self.reasoning_effort_combo.addItems(["low", "medium", "high"])
        self.reasoning_effort_combo.setCurrentText("medium")
        self.reasoning_effort_combo.setMinimumWidth(100)
        self.reasoning_effort_combo.setToolTip("Reasoning effort level for o-series models (o1, o3, o4)")
        model_row.addWidget(self.reasoning_effort_combo)
        
        model_row.addStretch()
        
        self.llm_status = QLabel("[READY]" if self.llm_enabled else "[NOT AVAILABLE]")
        self.llm_status.setStyleSheet(f"color: {'#3fb950' if self.llm_enabled else '#f85149'}; font-weight: bold;")
        model_row.addWidget(self.llm_status)
        llm_layout.addLayout(model_row)
        
        # Update reasoning effort visibility based on initial model
        self.on_model_changed(self.llm_model_combo.currentText())
        
        # Vision controls
        vision_frame = QFrame()
        vision_frame.setStyleSheet("QFrame { background: #161b22; border: 1px solid #30363d; padding: 8px; }")
        vision_layout = QVBoxLayout(vision_frame)
        vision_layout.setSpacing(8)
        vision_layout.setContentsMargins(8, 8, 8, 8)
        
        # Vision enable checkbox
        vision_header = QHBoxLayout()
        self.vision_enabled_checkbox = QCheckBox("Enable Vision (Send Images to LLM)")
        self.vision_enabled_checkbox.setChecked(True)
        self.vision_enabled_checkbox.setToolTip("Include camera images in LLM requests (requires vision-capable model)")
        self.vision_enabled_checkbox.setStyleSheet("font-weight: bold; color: #58a6ff;")
        vision_header.addWidget(self.vision_enabled_checkbox)
        vision_header.addStretch()
        vision_layout.addLayout(vision_header)
        
        # Camera source selection
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera:"))
        self.vision_camera_combo = QComboBox()
        self.vision_camera_combo.addItems(["Gripper Camera"])
        self.vision_camera_combo.setCurrentText("Gripper Camera")
        self.vision_camera_combo.setToolTip("Camera to send to LLM for vision")
        camera_row.addWidget(self.vision_camera_combo)
        
        # Image size selection
        camera_row.addWidget(QLabel("Resolution:"))
        self.vision_size_combo = QComboBox()
        self.vision_size_combo.addItems(["512x512", "768x768", "1024x1024", "Original"])
        self.vision_size_combo.setCurrentText("768x768")
        self.vision_size_combo.setToolTip("Image resolution sent to LLM (lower = faster/cheaper)")
        camera_row.addWidget(self.vision_size_combo)
        camera_row.addStretch()
        vision_layout.addLayout(camera_row)
        
        # Auto-update controls
        auto_row = QHBoxLayout()
        self.vision_auto_checkbox = QCheckBox("Auto-update")
        self.vision_auto_checkbox.setChecked(False)
        self.vision_auto_checkbox.setToolTip("Periodically send images to LLM for analysis")
        auto_row.addWidget(self.vision_auto_checkbox)
        
        auto_row.addWidget(QLabel("Every:"))
        self.vision_interval_spin = QSpinBox()
        self.vision_interval_spin.setRange(1, 60)  # Allow 1-60 seconds (was 5-60)
        self.vision_interval_spin.setValue(5)  # Default to 5 seconds (was 10)
        self.vision_interval_spin.setSuffix(" sec")
        self.vision_interval_spin.setEnabled(False)
        self.vision_auto_checkbox.toggled.connect(self.vision_interval_spin.setEnabled)
        self.vision_auto_checkbox.toggled.connect(self.toggle_vision_auto_update)
        auto_row.addWidget(self.vision_interval_spin)
        auto_row.addStretch()
        vision_layout.addLayout(auto_row)
        
        # Vision status indicator
        self.vision_status_label = QLabel("📷 Vision ready")
        self.vision_status_label.setStyleSheet("color: #58a6ff; font-size: 9pt;")
        vision_layout.addWidget(self.vision_status_label)
        
        llm_layout.addWidget(vision_frame)
        
        # Initialize vision auto-update timer
        self.vision_auto_timer = QTimer()
        self.vision_auto_timer.timeout.connect(self.vision_auto_update)
        
        # Command input
        llm_layout.addWidget(QLabel("Command:"))
        self.llm_command_input = QPlainTextEdit()
        self.llm_command_input.setPlaceholderText("Enter command (e.g., 'move arm to e2')")
        self.llm_command_input.setFixedHeight(80)
        llm_layout.addWidget(self.llm_command_input)
        
        # Execute button
        execute_btn = QPushButton("Execute")
        execute_btn.setEnabled(self.llm_enabled)
        execute_btn.clicked.connect(self._start_llm_execution_thread)
        execute_btn.setStyleSheet("""
            QPushButton {
                background: #238636;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background: #2ea043; }
            QPushButton:disabled { background: #21262d; color: #6e7681; }
        """)
        llm_layout.addWidget(execute_btn)
        
        # Execution logs (prominent during loop)
        exec_log_label = QLabel("Execution Log (live):")
        exec_log_label.setStyleSheet("color: #f85149; margin-top: 6px; font-weight: bold;")
        llm_layout.addWidget(exec_log_label)
        self.exec_log_display = QTextEdit()
        self.exec_log_display.setReadOnly(True)
        self.exec_log_display.setPlaceholderText("Execution logs will appear here (errors, warnings, motor read failures, overloads, IK issues).")
        self.exec_log_display.setStyleSheet("color: #c9d1d9;")
        self.exec_log_display.setMinimumHeight(180)
        llm_layout.addWidget(self.exec_log_display, 2)

        # Connect thread-safe log signals once the widget exists
        if not hasattr(self, "_exec_log_signals_connected"):
            self.exec_log_line_signal.connect(self._on_exec_log_line)
            self.exec_log_reset_signal.connect(self._on_exec_log_reset)
            self.execution_ui_mode_signal.connect(self._on_execution_ui_mode)
            self._exec_log_signals_connected = True

        # Response sections with scroll areas
        # Reasoning
        llm_layout.addWidget(QLabel("Reasoning:"))
        self.llm_reasoning_display = QTextEdit()
        self.llm_reasoning_display.setReadOnly(True)
        self.llm_reasoning_display.setPlaceholderText("LLM reasoning will appear here...")
        self.llm_reasoning_display.setStyleSheet("color: #f0883e;")
        llm_layout.addWidget(self.llm_reasoning_display, 1)
        
        # JSON Response (collapsible)
        response_label = QLabel("Full Response (JSON):")
        response_label.setStyleSheet("color: #8b949e; margin-top: 8px;")
        llm_layout.addWidget(response_label)
        
        self.llm_response_display = QTextEdit()
        self.llm_response_display.setReadOnly(True)
        self.llm_response_display.setPlaceholderText("JSON response will appear here...")
        self.llm_response_display.setStyleSheet("color: #8b949e;")
        self.llm_response_display.setMaximumHeight(150)
        llm_layout.addWidget(self.llm_response_display)
        
        # Action Preview
        action_label = QLabel("Validated Action:")
        action_label.setStyleSheet("color: #58a6ff; margin-top: 8px;")
        llm_layout.addWidget(action_label)
        
        self.llm_action_preview = QTextEdit()
        self.llm_action_preview.setReadOnly(True)
        self.llm_action_preview.setPlaceholderText("Validated action will appear here...")
        self.llm_action_preview.setStyleSheet("color: #58a6ff;")
        llm_layout.addWidget(self.llm_action_preview, 1)
        
        parent_layout.addWidget(llm_group)
        return llm_group
    
    def on_model_changed(self, model_name):
        """Show/hide reasoning effort selector based on model type."""
        # Clean model name (remove vision icon)
        clean_model = model_name.replace(" 👁️", "").strip()
        
        # Reasoning effort is only for o-series models
        is_reasoning_model = clean_model.startswith("o1") or clean_model.startswith("o3") or clean_model.startswith("o4")
        self.reasoning_effort_combo.setVisible(is_reasoning_model)
        # Also update the label visibility
        for i in range(self.reasoning_effort_combo.parent().layout().count()):
            item = self.reasoning_effort_combo.parent().layout().itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                if item.widget().text() == "Reasoning:":
                    item.widget().setVisible(is_reasoning_model)
        
        # Update vision status based on model capabilities
        vision_capable = "👁️" in model_name or any(x in clean_model.lower() for x in ['gpt-4o', 'gpt-4-turbo', 'vision', 'gpt-5', 'gpt-4.1'])
        if hasattr(self, 'vision_enabled_checkbox'):
            if not vision_capable and self.vision_enabled_checkbox.isChecked():
                self.vision_status_label.setText("⚠️ Selected model may not support vision")
                self.vision_status_label.setStyleSheet("color: #f0883e; font-size: 9pt;")
            elif vision_capable:
                self.vision_status_label.setText("📷 Vision ready (capable model)")
                self.vision_status_label.setStyleSheet("color: #3fb950; font-size: 9pt;")
            else:
                self.vision_status_label.setText("📷 Vision ready")
                self.vision_status_label.setStyleSheet("color: #58a6ff; font-size: 9pt;")

    def _format_exec_log_line(self, level: str, message: str) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        lvl = (level or "info").upper()
        return f"{ts} [{lvl}] {message}"

    def _clear_exec_log(self):
        self._exec_log_lines.clear()
        # UI updates must be in the main thread; use signal
        try:
            self.exec_log_reset_signal.emit()
        except Exception:
            pass

    def _append_exec_log(self, level: str, message: str):
        """Append a log line to the in-memory buffer and UI (thread-safe via Qt signals)."""
        line = self._format_exec_log_line(level, message)
        self._exec_log_lines.append(line)
        try:
            self.exec_log_line_signal.emit(line)
        except Exception:
            pass

    def _get_recent_exec_logs_for_prompt(self) -> str:
        """Return recent execution log lines for LLM context."""
        if not self._exec_log_lines:
            return ""
        return "\n".join(list(self._exec_log_lines)[-self._exec_log_prompt_max_lines :])

    def _set_execution_ui_mode(self, active: bool):
        """Make execution logs more prominent during goal-driven runs."""
        self._execution_active = bool(active)
        try:
            self.execution_ui_mode_signal.emit(bool(active))
        except Exception:
            pass

    def _on_exec_log_reset(self):
        """UI thread: clear execution log display."""
        try:
            if hasattr(self, "exec_log_display") and self.exec_log_display:
                self.exec_log_display.clear()
        except Exception:
            pass

    def _on_exec_log_line(self, line: str):
        """UI thread: append a single execution log line."""
        try:
            if not hasattr(self, "exec_log_display") or self.exec_log_display is None:
                return
            self.exec_log_display.append(line)
            if self.exec_log_display.document().blockCount() > self._exec_log_display_max_lines:
                self.exec_log_display.setPlainText("\n".join(list(self._exec_log_lines)[-self._exec_log_display_max_lines :]))
            cursor = self.exec_log_display.textCursor()
            cursor.movePosition(cursor.End)
            self.exec_log_display.setTextCursor(cursor)
        except Exception:
            pass

    def _on_execution_ui_mode(self, active: bool):
        """UI thread: toggle prominence of the log panel during execution."""
        try:
            if not hasattr(self, "exec_log_display") or self.exec_log_display is None:
                return
            if active:
                self.exec_log_display.setMinimumHeight(260)
                if hasattr(self, "llm_response_display") and self.llm_response_display:
                    self.llm_response_display.setMaximumHeight(110)
            else:
                self.exec_log_display.setMinimumHeight(180)
                if hasattr(self, "llm_response_display") and self.llm_response_display:
                    self.llm_response_display.setMaximumHeight(150)
        except Exception:
            pass
    
    def capture_camera_image(self, camera_source="main"):
        """Capture current frame from specified camera and encode to base64.
        
        Args:
            camera_source: "main", "gripper", or "both"
        
        Returns:
            str or list: Base64 encoded image(s)
        """
        try:
            # Get target resolution
            resolution = self.vision_size_combo.currentText()
            if resolution == "Original":
                target_size = None
            else:
                # Parse resolution (e.g., "768x768" -> (768, 768))
                w, h = map(int, resolution.split('x'))
                target_size = (w, h)
            
            # Capture from gripper camera (the only camera now)
            print(f"📸 Capturing image from gripper camera")
            
            if not self.gripper_camera:
                print(f"   ⚠️ Gripper camera object is None")
                return None
            if not self.gripper_camera.is_connected:
                print(f"   ⚠️ Gripper camera not connected")
                return None
            
            frame = self.gripper_camera.read()
            if frame is None:
                print(f"   ⚠️ Gripper camera returned None frame")
                return None
            
            original_shape = frame.shape
            
            # Resize if needed
            if target_size:
                frame = cv2.resize(frame, target_size)
            
            # Convert BGR to RGB for proper JPEG colors
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode to JPEG
            success, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if not success:
                print(f"   ⚠️ Gripper camera JPEG encoding failed")
                return None
            
            # Convert to base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            print(f"   ✅ Gripper camera: captured {original_shape} -> {frame.shape if target_size else 'original'}, {len(img_base64)//1024}KB")
            return img_base64
            
        except Exception as e:
            print(f"⚠️ Failed to capture camera image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def toggle_vision_auto_update(self, enabled):
        """Toggle periodic vision updates."""
        if enabled:
            interval_sec = self.vision_interval_spin.value()
            self.vision_auto_timer.start(interval_sec * 1000)
            self.vision_status_label.setText(f"📷 Auto-updating every {interval_sec}s")
            print(f"✅ Vision auto-update enabled ({interval_sec}s)")
        else:
            self.vision_auto_timer.stop()
            self.vision_status_label.setText("📷 Vision ready")
            print("⏸️ Vision auto-update disabled")
    
    def vision_auto_update(self):
        """Periodically analyze scene with LLM (auto-update mode)."""
        if not self.llm_enabled or not self.vision_enabled_checkbox.isChecked():
            return
        
        try:
            # Capture image
            camera_source = self.vision_camera_combo.currentText().split()[0].lower()  # "Main Camera" -> "main"
            image_data = self.capture_camera_image(camera_source)
            
            if image_data:
                # Build a scene analysis prompt
                prompt = "Analyze the current scene. What pieces are visible? What is the board state? Any recommendations?"
                
                # Send to LLM for analysis (without executing actions)
                self.vision_status_label.setText("📷 Analyzing scene...")
                self.vision_status_label.setStyleSheet("color: #f0883e; font-size: 9pt;")
                
                # Use simplified analysis (won't execute robot actions)
                self._analyze_scene_with_vision(prompt, image_data)
                
                self.vision_status_label.setText("📷 Auto-updating")
                self.vision_status_label.setStyleSheet("color: #3fb950; font-size: 9pt;")
            
        except Exception as e:
            print(f"⚠️ Vision auto-update error: {e}")
            self.vision_status_label.setText("⚠️ Vision update failed")
            self.vision_status_label.setStyleSheet("color: #f85149; font-size: 9pt;")
    
    def _analyze_scene_with_vision(self, prompt, image_data):
        """Send vision request to LLM for scene analysis (non-blocking)."""
        # This is a simplified version that just analyzes without robot control
        # The full implementation will be in execute_llm_command
        try:
            selected_model = self.llm_model_combo.currentText()
            
            # Build vision message
            messages = [
                {"role": "system", "content": "You are a chess assistant. Analyze the chess board and provide insights."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # Add image(s)
            if isinstance(image_data, list):
                # Multiple images
                for img in image_data:
                    if img:
                        messages[1]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                        })
            else:
                # Single image
                if image_data:
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
            
            # Send to LLM (simplified - just for analysis)
            # Use max_completion_tokens for o-series models, max_tokens for others
            is_reasoning_model = selected_model.startswith("o1") or selected_model.startswith("o3") or selected_model.startswith("o4")
            api_params = {"model": selected_model, "messages": messages}
            if is_reasoning_model:
                api_params["max_completion_tokens"] = 500
            else:
                api_params["max_tokens"] = 500
            response = self.llm_client.chat.completions.create(**api_params)
            
            analysis = response.choices[0].message.content
            print(f"🔍 Scene Analysis: {analysis[:200]}...")
            
        except Exception as e:
            print(f"⚠️ Scene analysis failed: {e}")
    
    def _quick_vision_analysis(self, prompt: str, image_data) -> Optional[str]:
        """Quick vision analysis for feedback loop - returns analysis text."""
        try:
            if not self.llm_enabled or not self.llm_client:
                return None
            
            selected_model = self.llm_model_combo.currentText().replace(" 👁️", "").strip()
            
            # Build vision message
            user_content = [{"type": "text", "text": prompt}]
            
            # Add image
            if isinstance(image_data, list):
                for img in image_data:
                    if img:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "low"}  # Low detail for speed
                        })
            else:
                if image_data:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "low"}
                    })
            
            messages = [
                {"role": "system", "content": "You are analyzing a chess board scene. Provide brief, actionable feedback about what you see."},
                {"role": "user", "content": user_content}
            ]
            
            # Quick analysis with lower token limit for speed
            # Use max_completion_tokens for o-series models, max_tokens for others
            is_reasoning_model = selected_model.startswith("o1") or selected_model.startswith("o3") or selected_model.startswith("o4")
            api_params = {"model": selected_model, "messages": messages}
            if is_reasoning_model:
                api_params["max_completion_tokens"] = 200
            else:
                api_params["max_tokens"] = 200
            response = self.llm_client.chat.completions.create(**api_params)
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ Quick vision analysis failed: {e}")
            return None

    def _llm_locate_pawn(self, image_b64: str) -> Optional[Dict[str, Any]]:
        """Locate a pawn (or any chess piece if pawn classification is uncertain) in the gripper camera image using the LLM.

        Returns JSON (normalized 0..1 coords):
        {
          "found": bool,
          "confidence": 0.0-1.0,
          "center_norm": {"x": 0..1, "y": 0..1},
          "bbox_norm": {"x1":0..1,"y1":0..1,"x2":0..1,"y2":0..1},
          "notes": str
        }
        """
        try:
            if not self.llm_enabled or not self.llm_client or not image_b64:
                return None

            selected_model = self.llm_model_combo.currentText().replace(" 👁️", "").strip()

            prompt = """You are a precise vision locator for a robot gripper camera.

Task: Find a chess piece in view. Prefer a PAWN if you can confidently identify one, but if you are unsure about piece type,
still return the best candidate chess piece rather than failing.

Important:
- Return coordinates NORMALIZED to image size: x and y are in [0,1]
- If there are multiple pieces, choose the piece closest to the image center.
- Only set found=false if there is clearly NO chess piece visible.

JSON schema:
{
  "found": true/false,
  "confidence": 0.0,
  "center_norm": {"x": 0.5, "y": 0.5},
  "bbox_norm": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
  "notes": "brief"
}
"""

            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}},
            ]
            messages = [
                {"role": "system", "content": "You are a vision locator. Output only valid JSON."},
                {"role": "user", "content": user_content},
            ]

            # Model-appropriate parameters (gpt-5 and o-series require max_completion_tokens)
            is_o_series = selected_model.startswith("o")
            is_gpt5 = selected_model.startswith("gpt-5")
            api_params = {"model": selected_model, "messages": messages}
            if is_o_series or is_gpt5:
                api_params["max_completion_tokens"] = 300
            else:
                api_params["max_tokens"] = 300

            # Structured JSON output where supported (not for o-series)
            if not is_o_series:
                api_params["response_format"] = {"type": "json_object"}
                if not is_gpt5:
                    api_params["temperature"] = 0.0

            resp = self.llm_client.chat.completions.create(**api_params)
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ Pawn localization failed: {e}")
            return None

    def _llm_locate_chessboard(self, image_b64: str) -> Optional[Dict[str, Any]]:
        """Locate the chessboard in the gripper camera image using the LLM.

        Returns JSON (normalized 0..1 coords):
        {
          "found": bool,
          "confidence": 0.0-1.0,
          "center_norm": {"x": 0..1, "y": 0..1},
          "bbox_norm": {"x1":0..1,"y1":0..1,"x2":0..1,"y2":0..1},
          "angle_deg": <float or null>,   // optional: board rotation in image coordinates
          "notes": str
        }
        """
        try:
            if not self.llm_enabled or not self.llm_client or not image_b64:
                return None

            selected_model = self.llm_model_combo.currentText().replace(" 👁️", "").strip()

            prompt = """You are a precise vision locator for a robot gripper camera.

Task: Locate the CHESSBOARD (the 8x8 board) in the image.

Important:
- Return coordinates NORMALIZED to image size: x and y are in [0,1]
- If the board is partially visible, still return the best estimate if you are reasonably confident.
- Only set found=false if there is clearly NO chessboard visible.

JSON schema:
{
  "found": true/false,
  "confidence": 0.0,
  "center_norm": {"x": 0.5, "y": 0.5},
  "bbox_norm": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
  "angle_deg": null,
  "notes": "brief"
}
"""

            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}},
            ]
            messages = [
                {"role": "system", "content": "You are a vision locator. Output only valid JSON."},
                {"role": "user", "content": user_content},
            ]

            # Model-appropriate parameters (gpt-5 and o-series require max_completion_tokens)
            is_o_series = selected_model.startswith("o")
            is_gpt5 = selected_model.startswith("gpt-5")
            api_params = {"model": selected_model, "messages": messages}
            if is_o_series or is_gpt5:
                api_params["max_completion_tokens"] = 350
            else:
                api_params["max_tokens"] = 350

            # Structured JSON output where supported (not for o-series)
            if not is_o_series:
                api_params["response_format"] = {"type": "json_object"}
                if not is_gpt5:
                    api_params["temperature"] = 0.0

            resp = self.llm_client.chat.completions.create(**api_params)
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ Chessboard localization failed: {e}")
            return None

    def _set_gripper(self, percent: float):
        """Set gripper opening percentage (0=open, 100=closed)."""
        try:
            if "gripper" not in self.all_motors:
                return
            val = max(0.0, min(100.0, float(percent)))
            self.bus.write("Goal_Position", "gripper", val, normalize=True)
            time.sleep(0.25)
        except Exception as e:
            print(f"⚠️ Gripper command failed: {e}")

    def _auto_pick_pawn(self, thread: Optional["LLMExecutionThread"] = None) -> Tuple[bool, str]:
        """Closed-loop routine: overview -> search -> center -> descend -> close -> lift.

        Uses ONLY the gripper camera for perception. The LLM is used only to localize the pawn in the image.
        """
        def emit_status(msg: str):
            if thread:
                thread.status_update.emit(msg)
            else:
                self.status_bar.setText(msg)

        def emit_reasoning(msg: str):
            if thread:
                thread.reasoning_update.emit(msg)
            else:
                self.llm_reasoning_display.setText(msg)

        def emit_response(msg: str):
            if thread:
                thread.response_update.emit(msg)
            else:
                self.llm_response_display.setText(msg)

        if self.kinematics is None:
            return False, "Kinematics not available (needed for safe Cartesian moves)"

        emit_status("🧭 Auto pick: moving to overview...")
        emit_reasoning("Auto pick pawn: overview → search → center → descend → close → lift\n")

        # Open gripper first
        self._set_gripper(0.0)

        # Helper: try to find a saved pose by fuzzy name
        def _get_saved_pose(name_candidates: List[str]) -> Optional[Dict[str, float]]:
            saved = self._load_saved_positions() or {}
            if not saved:
                return None
            # map lower->actual key
            key_map = {str(k).strip().lower(): k for k in saved.keys()}
            for cand in name_candidates:
                k = key_map.get(cand.strip().lower())
                if k and isinstance(saved.get(k), dict):
                    pos = saved[k].get("positions")
                    if isinstance(pos, dict) and pos:
                        return {m: float(v) for m, v in pos.items() if m in self.all_motors}
            return None

        # Helper: smooth joint-space move to a target pose in small chunks (avoids one big clamp)
        def _goto_pose(target: Dict[str, float], steps: int = 4, settle_s: float = 0.7) -> bool:
            try:
                self.update_position_model()
                cur = dict(self.position_model.get("joints", {}))
                for s in range(1, steps + 1):
                    if thread and not thread.running:
                        return False
                    alpha = s / steps
                    inter_action = {}
                    for m, tgt in target.items():
                        if m not in cur:
                            continue
                        inter_action[f"{m}.pos"] = float(cur[m]) + (float(tgt) - float(cur[m])) * alpha
                    safe = self._validate_llm_action(inter_action, cur, is_sequence_step=True)
                    if safe:
                        self._execute_llm_action(safe)
                        time.sleep(settle_s)
                        self.update_position_model()
                        cur = dict(self.position_model.get("joints", {}))
                return True
            except Exception as e:
                print(f"⚠️ goto_pose failed: {e}")
                return False

        # Go to a safe overview pose (prefer user's saved Overview/Ready)
        self.update_position_model()
        current_positions = dict(self.position_model.get("joints", {}))
        saved_overview = _get_saved_pose(["overview", "Overview", "OVERVIEW", "ready", "Ready"])
        if saved_overview:
            emit_status("🧭 Auto pick: going to saved OVERVIEW/READY pose...")
            _goto_pose(saved_overview, steps=5, settle_s=0.7)
        else:
            # fallback: generic overview
            overview_action = {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 40.0,
                "elbow_flex.pos": 40.0,
                "wrist_flex.pos": -30.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 0.0,
            }
            safe_overview = self._validate_llm_action(overview_action, current_positions, is_sequence_step=True)
            if safe_overview:
                self._execute_llm_action(safe_overview)
                time.sleep(0.7)

        # Search by sweeping shoulder_pan a bit until pawn is visible
        emit_status("🔍 Auto pick: searching for pawn...")
        # IMPORTANT: sweep is PAN-ONLY to avoid changing height while searching (prevents board hits).
        # We keep the rest of the joints at the current "overview" pose.
        sweep_angles = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 60, 45, 30, 15, 0, -15, -30, -45, -60]
        last_obs = None
        for ang in sweep_angles:
            if thread and not thread.running:
                return False, "Cancelled"
            # Small joint motion for search
            self.update_position_model()
            cur = dict(self.position_model.get("joints", {}))
            act = {"shoulder_pan.pos": float(ang)}
            safe = self._validate_llm_action(act, cur)
            if safe:
                self._execute_llm_action(safe)
                time.sleep(0.8)  # give camera time to settle (reduces motion blur)

            # Sample multiple frames per angle; accept if ANY sees the piece
            best = None
            for _ in range(2):
                img = self.capture_camera_image("gripper")
                obs = self._llm_locate_pawn(img) if img else None
                if obs:
                    last_obs = obs
                    if best is None or float(obs.get("confidence", 0.0)) > float(best.get("confidence", 0.0)):
                        best = obs
                time.sleep(0.15)

            emit_response(json.dumps(best or last_obs or {"found": False}, indent=2))
            if best and best.get("found") and float(best.get("confidence", 0.0)) >= 0.35:
                last_obs = best
                break
        else:
            return False, "Pawn not found in search sweep"

        # Calibrate a simple image Jacobian for (dx,dy)->(image error) using tiny probe moves
        def get_err(obs: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
            try:
                c = obs.get("center_norm") or {}
                bx = obs.get("bbox_norm") or {}
                ex = float(c.get("x", 0.5)) - 0.5
                ey = float(c.get("y", 0.5)) - 0.5
                area = max(0.0, float(bx.get("x2", 0.0)) - float(bx.get("x1", 0.0))) * max(
                    0.0, float(bx.get("y2", 0.0)) - float(bx.get("y1", 0.0))
                )
                return ex, ey, area
            except Exception:
                return None

        emit_status("🎯 Auto pick: centering pawn...")
        base_img = self.capture_camera_image("gripper")
        base_obs = self._llm_locate_pawn(base_img) if base_img else None
        if not base_obs or not base_obs.get("found"):
            base_obs = last_obs
        if not base_obs or not base_obs.get("found"):
            return False, "Lost pawn before centering"

        base_err = get_err(base_obs)
        if not base_err:
            return False, "Pawn localization malformed"

        # Default J (fallback): assume dy affects x-error, dx affects y-error
        J = np.array([[0.0, 0.015], [0.015, 0.0]], dtype=float)  # error per mm (rough)
        dx_probe = 6.0
        dy_probe = 6.0
        try:
            # Probe +dx
            self._move_gripper_mm(dx_probe, 0.0, 0.0, apply_ui_step=False)
            img1 = self.capture_camera_image("gripper")
            obs1 = self._llm_locate_pawn(img1) if img1 else None
            self._move_gripper_mm(-dx_probe, 0.0, 0.0, apply_ui_step=False)
            # Probe +dy
            self._move_gripper_mm(0.0, dy_probe, 0.0, apply_ui_step=False)
            img2 = self.capture_camera_image("gripper")
            obs2 = self._llm_locate_pawn(img2) if img2 else None
            self._move_gripper_mm(0.0, -dy_probe, 0.0, apply_ui_step=False)

            e0 = get_err(base_obs)
            e1 = get_err(obs1) if obs1 and obs1.get("found") else None
            e2 = get_err(obs2) if obs2 and obs2.get("found") else None
            if e0 and e1 and e2:
                J = np.array(
                    [
                        [(e1[0] - e0[0]) / dx_probe, (e2[0] - e0[0]) / dy_probe],
                        [(e1[1] - e0[1]) / dx_probe, (e2[1] - e0[1]) / dy_probe],
                    ],
                    dtype=float,
                )
        except Exception as e:
            print(f"⚠️ Jacobian probe failed, using fallback mapping: {e}")

        # Servo loop: center in XY, then descend in Z while re-centering
        def pinv2(m: np.ndarray) -> np.ndarray:
            return np.linalg.pinv(m)

        Jinv = pinv2(J)
        center_tol = 0.06
        max_center_iters = 10

        for i in range(max_center_iters):
            if thread and not thread.running:
                return False, "Cancelled"
            img = self.capture_camera_image("gripper")
            obs = self._llm_locate_pawn(img) if img else None
            if not obs or not obs.get("found"):
                emit_status("⚠️ Auto pick: lost pawn during centering, re-searching...")
                return False, "Lost pawn during centering"
            emit_response(json.dumps(obs, indent=2))
            ex, ey, area = get_err(obs) or (0.0, 0.0, 0.0)
            emit_status(f"🎯 Centering: err=({ex:+.3f},{ey:+.3f}) area={area:.3f}")
            if abs(ex) < center_tol and abs(ey) < center_tol:
                break

            e = np.array([ex, ey], dtype=float)
            # Gain: smaller steps when close (area grows as we get closer)
            gain = 0.65 if area < 0.02 else 0.4
            dxy = -gain * (Jinv @ e)
            dx_cmd = float(np.clip(dxy[0] * 100.0, -10.0, 10.0))  # scale to mm
            dy_cmd = float(np.clip(dxy[1] * 100.0, -10.0, 10.0))
            self._move_gripper_mm(dx_cmd, dy_cmd, 0.0, apply_ui_step=False)

        # Approach: descend in small increments while keeping centered
        emit_status("⬇️ Auto pick: approaching pawn...")
        max_descend_steps = 8
        for step in range(max_descend_steps):
            if thread and not thread.running:
                return False, "Cancelled"
            img = self.capture_camera_image("gripper")
            obs = self._llm_locate_pawn(img) if img else None
            if not obs or not obs.get("found"):
                return False, "Lost pawn during approach"
            emit_response(json.dumps(obs, indent=2))
            ex, ey, area = get_err(obs) or (0.0, 0.0, 0.0)

            # Re-center a bit
            if abs(ex) > center_tol or abs(ey) > center_tol:
                e = np.array([ex, ey], dtype=float)
                dxy = -(0.35 * (Jinv @ e))
                dx_cmd = float(np.clip(dxy[0] * 100.0, -6.0, 6.0))
                dy_cmd = float(np.clip(dxy[1] * 100.0, -6.0, 6.0))
                self._move_gripper_mm(dx_cmd, dy_cmd, 0.0, apply_ui_step=False)

            # If the pawn is "big enough" in view, start grasp
            if area >= 0.035:
                break

            # Descend a little (dz clamp inside move)
            self._move_gripper_mm(0.0, 0.0, -5.0, apply_ui_step=False)

        # Final grasp
        emit_status("🤏 Auto pick: grasping...")
        self._move_gripper_mm(0.0, 0.0, -4.0, apply_ui_step=False)
        self._set_gripper(85.0)
        time.sleep(0.4)
        emit_status("⬆️ Auto pick: lifting...")
        self._move_gripper_mm(0.0, 0.0, +25.0, apply_ui_step=False)
        time.sleep(0.6)

        return True, "Auto pick completed (check gripper for pawn)"
    
    def _start_llm_execution_thread(self):
        """Start LLM execution in background thread to prevent UI blocking."""
        if not self.llm_enabled or not self.llm_client:
            self.status_bar.setText("❌ LLM not available")
            return
        
        command = self.llm_command_input.toPlainText().strip()
        if not command:
            self.status_bar.setText("⚠️ Please enter a command")
            return
        
        # Stop any existing execution thread
        if hasattr(self, 'execution_thread') and self.execution_thread and self.execution_thread.isRunning():
            self.execution_thread.running = False
            self.execution_thread.wait(1000)  # Wait up to 1 second
        
        # Disable execute button during execution
        self.llm_command_input.setEnabled(False)
        
        # Initialize UI
        self._clear_exec_log()
        self._set_execution_ui_mode(True)
        self._append_exec_log("info", f"Starting execution: {command}")
        self.status_bar.setText("🎯 Starting goal-driven execution...")
        self.llm_response_display.setText("Initializing...")
        self.llm_reasoning_display.setText(f"Goal: {command}\n\nStarting iterative execution...\n")
        self.llm_action_preview.setText("")
        
        # Create and start execution thread
        self.execution_thread = LLMExecutionThread(self, command)
        self.execution_thread.status_update.connect(self.status_bar.setText)
        self.execution_thread.reasoning_update.connect(self.llm_reasoning_display.setText)
        self.execution_thread.response_update.connect(self.llm_response_display.setText)
        self.execution_thread.action_preview_update.connect(self.llm_action_preview.setText)
        self.execution_thread.finished_signal.connect(self._on_execution_finished)
        self.execution_thread.start()
    
    def _on_execution_finished(self, success: bool, message: str):
        """Called when execution thread finishes."""
        self.llm_command_input.setEnabled(True)  # Re-enable input
        self._set_execution_ui_mode(False)
        self._append_exec_log("info" if success else "error", f"Execution finished: {message}")
        if not success:
            self.status_bar.setText(f"❌ {message}")
    
    def execute_llm_command(self):
        """Backward compatibility - redirects to thread-based execution."""
        self._start_llm_execution_thread()
    
    def _execute_llm_command_internal(self, command: str, thread: Optional[LLMExecutionThread] = None):
        """Internal execution method - can be called from thread or directly."""
        # Helper to emit updates (works with or without thread)
        def emit_status(msg):
            if thread:
                thread.status_update.emit(msg)
            else:
                self.status_bar.setText(msg)
        
        def emit_reasoning(msg):
            if thread:
                thread.reasoning_update.emit(msg)
            else:
                self.llm_reasoning_display.setText(msg)
        
        def emit_response(msg):
            if thread:
                thread.response_update.emit(msg)
            else:
                self.llm_response_display.setText(msg)
        
        def emit_action_preview(msg):
            if thread:
                thread.action_preview_update.emit(msg)
            else:
                self.llm_action_preview.setText(msg)
        
        if not self.llm_enabled or not self.llm_client:
            emit_status("❌ LLM not available")
            if thread:
                thread.finished_signal.emit(False, "LLM not available")
            return
        
        # Initialize goal-driven loop
        self._clear_exec_log()
        self._set_execution_ui_mode(True)
        self._append_exec_log("info", f"Execution started: {command}")
        emit_status("🎯 Starting goal-driven execution...")
        emit_response("Initializing...")
        emit_reasoning(f"Goal: {command}\n\nStarting iterative execution...\n")
        emit_action_preview("")
        
        # PAUSE monitoring thread to avoid port conflicts during command execution
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            self.monitor_thread.paused = True
            time.sleep(0.2)  # Wait for current read to complete
            print("⏸️ Monitoring paused for command execution")
            self._append_exec_log("info", "Monitoring paused to avoid port conflicts")
        
        # Track iteration history
        iteration_history = []
        max_iterations = 25  # Increased to allow more attempts to reach goal
        iteration = 0
        consecutive_failures = 0  # Track repeated failures to detect stuck state
        last_action_hash = None  # Track if we're repeating the same action
        movement_history = []  # Track recent movements to detect repeated large movements
        
        try:
            # Special deterministic routines (avoid open-ended LLM joint-guessing)
            cmd = (command or "").strip().lower()
            if cmd in {"pick pawn", "pickup pawn", "pick up pawn", "find pawn and pick it up"}:
                emit_status("🤖 Running auto pick routine...")
                ok, msg = self._auto_pick_pawn(thread=thread)
                if ok:
                    emit_status(f"✅ {msg}")
                    if thread:
                        thread.finished_signal.emit(True, msg)
                else:
                    emit_status(f"❌ {msg}")
                    if thread:
                        thread.finished_signal.emit(False, msg)
                return

            while iteration < max_iterations:
                if thread and not thread.running:
                    break  # Thread was stopped
                
                iteration += 1
                emit_status(f"🔄 Iteration {iteration}/{max_iterations}: Planning next action...")
                
                # Get current robot state
                self.update_position_model()
                current_positions = {}
                for motor_name in self.all_motors.keys():
                    try:
                        pos = self.bus.read("Present_Position", motor_name, normalize=True)
                        current_positions[motor_name] = pos
                    except:
                        current_positions[motor_name] = self.position_model["joints"].get(motor_name, 0)
                        self._append_exec_log("warning", f"Read Present_Position failed for {motor_name}; using last-known value")
                
                # Capture current image
                image_data = None
                if self.vision_enabled_checkbox.isChecked():
                    camera_source = self.vision_camera_combo.currentText().split()[0].lower()
                    image_data = self.capture_camera_image(camera_source)
                
                # Build context-aware prompt with iteration history
                iteration_context = self._build_iteration_context(command, current_positions, iteration, iteration_history)
                
                # Get LLM's next action
                llm_output = self._get_llm_action(iteration_context, image_data, current_positions)
                
                if not llm_output:
                    emit_status(f"❌ Iteration {iteration}: LLM failed to respond")
                    if thread:
                        thread.finished_signal.emit(False, f"Iteration {iteration}: LLM failed to respond")
                    break
                
                # Display what LLM wants to do
                explanation = llm_output.get("explanation", "")
                # Get current reasoning text
                if thread:
                    # For thread, we need to append - store in a variable that accumulates
                    if not hasattr(thread, '_reasoning_text'):
                        thread._reasoning_text = f"Goal: {command}\n\nStarting iterative execution...\n"
                    thread._reasoning_text += f"\n\n--- Iteration {iteration} ---\n{explanation}\n"
                    emit_reasoning(thread._reasoning_text)
                else:
                    current_reasoning = self.llm_reasoning_display.toPlainText()
                    emit_reasoning(current_reasoning + f"\n\n--- Iteration {iteration} ---\n{explanation}\n")
                
                # Display full response
                emit_response(json.dumps(llm_output, indent=2))
                
                # Execute the action
                sequence = llm_output.get("sequence", None)
                action = llm_output.get("action", {})
                
                executed = False
                execution_error = None
                
                try:
                    if sequence:
                        executed = self._execute_llm_sequence(sequence, current_positions, original_command=command)
                        if not executed:
                            execution_error = "Sequence stopped (likely due to overload)"
                    elif action:
                        safe_action = self._validate_llm_action(action, current_positions)
                        if safe_action:
                            self._execute_llm_action(safe_action)
                            executed = True
                        else:
                            execution_error = "Action failed validation"
                except RuntimeError as e:
                    # Overload error from _execute_llm_action
                    if "overload" in str(e).lower():
                        execution_error = "Motor overload detected"
                        executed = False
                    else:
                        raise
                
                if not executed:
                    error_msg = execution_error or "Action execution failed"
                    emit_status(f"⚠️ Iteration {iteration}: {error_msg}")
                    iteration_history.append({"iteration": iteration, "status": "execution_failed", "action": llm_output, "error": error_msg})
                    consecutive_failures += 1
                    
                    # If overload detected, add extra cool-down pause - this is critical for motor safety
                    if "overload" in error_msg.lower():
                        emit_status(f"⏸️ OVERLOAD detected - pausing 5 seconds before retry...")
                        time.sleep(5.0)  # Increased for motor safety
                    
                    # Stop if too many consecutive failures
                    if consecutive_failures >= 3:
                        emit_status(f"❌ Stopped: {consecutive_failures} consecutive failures")
                        current_reasoning = thread._reasoning_text if thread and hasattr(thread, '_reasoning_text') else ""
                        emit_reasoning(
                            current_reasoning + 
                            f"\n\n❌ Stopped after {consecutive_failures} consecutive failures. Robot may be stuck or overloaded."
                        )
                        if thread:
                            thread.finished_signal.emit(False, f"{consecutive_failures} consecutive failures")
                        break
                    continue
                else:
                    consecutive_failures = 0  # Reset on success
                
                # Wait for action to complete
                time.sleep(1.0)
                
                # Check if positions actually changed (stuck detection)
                new_positions = {}
                for motor_name in self.all_motors.keys():
                    try:
                        pos = self.bus.read("Present_Position", motor_name, normalize=True)
                        new_positions[motor_name] = pos
                    except:
                        new_positions[motor_name] = current_positions.get(motor_name, 0)
                
                # Calculate position change
                total_change = sum(abs(new_positions.get(m, 0) - current_positions.get(m, 0)) 
                                  for m in self.all_motors.keys())
                
                # Track movement history (last 3 iterations)
                movement_history.append(total_change)
                if len(movement_history) > 3:
                    movement_history.pop(0)
                
                # Detect repeated large movements (could cause overheating)
                if len(movement_history) >= 2:
                    # Only pause for very large movements (increased threshold)
                    recent_large_movements = sum(1 for m in movement_history[-3:] if m > 30.0)
                    if recent_large_movements >= 3:
                        emit_status(f"⚠️ Multiple large movements detected - brief pause...")
                        current_reasoning = thread._reasoning_text if thread and hasattr(thread, '_reasoning_text') else ""
                        emit_reasoning(
                            current_reasoning + 
                            f"\n⚠️ Multiple large movements detected - brief pause for motor safety\n"
                        )
                        time.sleep(0.5)  # Reduced cool-down pause
                
                if total_change < 2.0:  # Less than 2 degrees total change = stuck
                    consecutive_failures += 1
                    current_reasoning = thread._reasoning_text if thread and hasattr(thread, '_reasoning_text') else ""
                    emit_reasoning(
                        current_reasoning + 
                        f"\n⚠️ Robot barely moved (Δ={total_change:.1f}°) - may be stuck or overloaded\n"
                    )
                
                # Check if goal is reached
                goal_reached, completion_feedback = self._check_goal_completion(
                    command, new_positions, image_data, action_executed=True, position_change=total_change  # Use updated positions
                )
                
                # Record iteration
                iteration_history.append({
                    "iteration": iteration,
                    "action": llm_output,
                    "goal_reached": goal_reached,
                    "feedback": completion_feedback,
                    "position_change": total_change
                })
                
                # Update display
                current_reasoning = thread._reasoning_text if thread and hasattr(thread, '_reasoning_text') else ""
                emit_reasoning(
                    current_reasoning + 
                    f"\n✅ Action executed (moved {total_change:.1f}°)\n🎯 Goal status: {'REACHED' if goal_reached else 'NOT YET REACHED'}\n"
                    f"Feedback: {completion_feedback}\n"
                )
                if thread and hasattr(thread, '_reasoning_text'):
                    thread._reasoning_text = current_reasoning + f"\n✅ Action executed (moved {total_change:.1f}°)\n🎯 Goal status: {'REACHED' if goal_reached else 'NOT YET REACHED'}\nFeedback: {completion_feedback}\n"
                
                if goal_reached:
                    emit_status(f"✅ Goal achieved in {iteration} iteration(s)!")
                    current_reasoning = thread._reasoning_text if thread and hasattr(thread, '_reasoning_text') else ""
                    emit_reasoning(
                        current_reasoning + 
                        f"\n\n🎉 SUCCESS! Goal '{command}' has been achieved!\n"
                    )
                    if thread:
                        thread.finished_signal.emit(True, f"Goal achieved in {iteration} iteration(s)")
                    break
                else:
                    # Continue loop - LLM will plan next action
                    emit_status(f"🔄 Iteration {iteration} complete. Goal not yet reached. Planning next action...")
                    # Add delay between iterations to prevent motor overheating
                    # Longer delay if we just made a large movement
                    if total_change > 30.0:
                        time.sleep(0.8)  # Brief pause after large movements
                    else:
                        time.sleep(0.5)  # Reduced pause between iterations
            
            if iteration >= max_iterations:
                emit_status(f"⚠️ Reached max iterations ({max_iterations}). Goal may not be fully achieved.")
                current_reasoning = thread._reasoning_text if thread and hasattr(thread, '_reasoning_text') else ""
                emit_reasoning(
                    current_reasoning + 
                    f"\n\n⚠️ Stopped after {max_iterations} iterations. Goal may require more steps."
                )
                if thread:
                    thread.finished_signal.emit(False, f"Reached max iterations ({max_iterations})")
                
        except Exception as e:
            emit_status(f"❌ Goal-driven execution error: {str(e)[:50]}...")
            current_reasoning = thread._reasoning_text if thread and hasattr(thread, '_reasoning_text') else ""
            emit_reasoning(
                current_reasoning + 
                f"\n\n❌ Error: {str(e)}"
            )
            import traceback
            traceback.print_exc()
            if thread:
                thread.finished_signal.emit(False, f"Error: {str(e)}")
        finally:
            # RESUME monitoring thread after command execution
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                self.monitor_thread.paused = False
                print("▶️ Monitoring resumed")
                self._append_exec_log("info", "Monitoring resumed")
            self._set_execution_ui_mode(False)
    
    def _build_iteration_context(self, original_command: str, current_positions: Dict[str, float], 
                                  iteration: int, history: List[Dict]) -> str:
        """Build context for LLM including iteration history and current state."""
        # Get current square if available
        current_square = None
        if "shoulder_pan" in current_positions and "shoulder_lift" in current_positions:
            current_square = self.calculate_robot_square(
                current_positions["shoulder_pan"],
                current_positions["shoulder_lift"]
            )
        
        # Get board info
        board_info = self.get_chess_board_info()
        board_context = ""
        if board_info:
            board_context = f"\nChess Board: Calibrated, current square: {current_square if current_square else 'unknown'}"
        
        # Build history summary
        history_text = ""
        if history:
            history_text = "\n\nPrevious iterations:\n"
            for h in history[-3:]:  # Last 3 iterations
                hist_iter = h.get("iteration", 0)
                hist_feedback = h.get("feedback", "")
                history_text += f"  Iteration {hist_iter}: {hist_feedback[:100]}...\n"
        
        context = f"""GOAL-DRIVEN EXECUTION - Iteration {iteration}

ORIGINAL GOAL: {original_command}

CURRENT STATE:
- Joint positions: {json.dumps(current_positions, indent=2)}
- Current square: {current_square if current_square else "unknown"}
{board_context}
{history_text}

RECENT EXECUTION LOGS (critical, read carefully):
{self._get_recent_exec_logs_for_prompt() if self._get_recent_exec_logs_for_prompt() else "(none)"}

YOUR TASK:
Based on the goal and current state, determine what action to take NEXT.
- If goal is not yet achieved, plan the next step toward the goal
- If you're close but not quite there, make a small adjustment
- If you need to see more, move to overview position first
- Keep actions incremental and safe

Output either a single action or a short sequence (1-3 steps max per iteration).
"""
        return context
    
    def _get_llm_action(self, context: str, image_data, current_positions: Dict[str, float]) -> Optional[Dict]:
        """Get next action from LLM given context."""
        try:
            selected_model = self.llm_model_combo.currentText().replace(" 👁️", "").strip()
            
            # Build prompt with context
            base_prompt = self._build_llm_prompt("", current_positions)  # Get base prompt structure
            # Replace the command part with our iteration context
            full_prompt = base_prompt.replace('User command: ""', f'Iteration Context:\n{context}')
            
            # Build messages
            if image_data:
                # Count and log images being sent
                if isinstance(image_data, list):
                    valid_images = [img for img in image_data if img]
                    img_sizes = [len(img) // 1024 for img in valid_images]  # Size in KB
                    print(f"📷 Sending {len(valid_images)} image(s) to LLM: {img_sizes} KB each")
                else:
                    img_size = len(image_data) // 1024
                    print(f"📷 Sending 1 image to LLM: {img_size} KB")
                
                user_content = [{"type": "text", "text": full_prompt}]
                if isinstance(image_data, list):
                    for i, img in enumerate(image_data):
                        if img:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}
                            })
                            print(f"   📎 Image {i+1} attached (base64 length: {len(img)})")
                else:
                    if image_data:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "high"}
                        })
                        print(f"   📎 Image attached (base64 length: {len(image_data)})")
                
                messages = [
                    {"role": "system", "content": "You are a robot control assistant. Analyze the current state and plan the next action toward the goal. Output only valid JSON."},
                    {"role": "user", "content": user_content}
                ]
            else:
                print("📷 No image data - sending text-only request to LLM")
                messages = [
                    {"role": "system", "content": "You are a robot control assistant. Analyze the current state and plan the next action toward the goal. Output only valid JSON."},
                    {"role": "user", "content": full_prompt}
                ]
            
            # Call LLM
            is_reasoning_model = selected_model.startswith("o1") or selected_model.startswith("o3") or selected_model.startswith("o4")
            
            api_params = {
                "model": selected_model,
                "messages": messages
            }
            
            if not is_reasoning_model:
                api_params["response_format"] = {"type": "json_object"}
                if not selected_model.startswith("gpt-5"):
                    api_params["temperature"] = 0.3
            else:
                api_params["reasoning_effort"] = self.reasoning_effort_combo.currentText()
            
            response = self.llm_client.chat.completions.create(**api_params)
            response_content = response.choices[0].message.content
            return json.loads(response_content)
            
        except Exception as e:
            print(f"⚠️ LLM action request failed: {e}")
            return None
    
    def _check_goal_completion(self, command: str, current_positions: Dict[str, float], 
                                image_data, action_executed: bool, position_change: float) -> Tuple[bool, str]:
        """Goal completion judge - determines if goal has been reached.
        
        Returns: (goal_reached: bool, feedback: str)
        """
        try:
            # Update position model
            self.update_position_model()
            
            # If nothing was executed, don't declare success
            if not action_executed:
                return False, "No action executed yet"
            
            # If almost no movement, don't declare success
            if position_change < 1.0:
                return False, f"Insufficient movement (Δ={position_change:.1f}°); continuing"
            
            # Get current square
            current_square = None
            if "shoulder_pan" in current_positions and "shoulder_lift" in current_positions:
                current_square = self.calculate_robot_square(
                    current_positions["shoulder_pan"],
                    current_positions["shoulder_lift"]
                )
            
            # Use LLM as goal completion judge if vision is available
            if image_data and self.llm_enabled:
                return self._llm_goal_judge(command, current_positions, current_square, image_data)
            else:
                # Rule-based judge (fallback)
                return self._rule_based_goal_judge(command, current_positions, current_square)
                
        except Exception as e:
            print(f"⚠️ Goal completion check failed: {e}")
            return False, f"Error checking goal: {e}"
    
    def _llm_goal_judge(self, command: str, current_positions: Dict[str, float], 
                        current_square: Optional[str], image_data) -> Tuple[bool, str]:
        """LLM-based goal completion judge - analyzes scene to determine if goal is reached."""
        try:
            selected_model = self.llm_model_combo.currentText().replace(" 👁️", "").strip()
            
            # Build judge prompt
            judge_prompt = f"""GOAL COMPLETION JUDGE

Original goal: {command}

Current robot state:
- Joint positions: {json.dumps(current_positions, indent=2)}
- Current square: {current_square if current_square else "unknown"}

Analyze the image and determine if the goal has been achieved.

Output JSON:
{{
  "goal_reached": <true if goal is achieved, false if not>,
  "confidence": <0.0 to 1.0, how confident you are>,
  "feedback": "<brief explanation of current state relative to goal>",
  "whats_missing": "<if not reached, what still needs to be done>"
}}

Examples:
- Goal: "find a1" → reached if gripper is over a1 square
- Goal: "touch a1" → reached if gripper is touching a1
- Goal: "pick up piece" → reached if piece is in gripper
"""
            
            user_content = [{"type": "text", "text": judge_prompt}]
            
            if isinstance(image_data, list):
                for img in image_data:
                    if img:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}
                        })
            else:
                if image_data:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "high"}
                    })
            
            messages = [
                {"role": "system", "content": "You are a goal completion judge. Analyze if the robot has achieved its goal. Output only valid JSON."},
                {"role": "user", "content": user_content}
            ]
            
            is_reasoning_model = selected_model.startswith("o1") or selected_model.startswith("o3") or selected_model.startswith("o4")
            
            api_params = {
                "model": selected_model,
                "messages": messages
            }
            
            if not is_reasoning_model:
                api_params["response_format"] = {"type": "json_object"}
                if not selected_model.startswith("gpt-5"):
                    api_params["temperature"] = 0.2  # Lower temp for more consistent judging
            else:
                api_params["reasoning_effort"] = "low"  # Faster for judging
            
            response = self.llm_client.chat.completions.create(**api_params)
            result = json.loads(response.choices[0].message.content)
            
            goal_reached = result.get("goal_reached", False)
            feedback = result.get("feedback", "No feedback")
            whats_missing = result.get("whats_missing", "")
            
            full_feedback = feedback
            if whats_missing and not goal_reached:
                full_feedback += f" Missing: {whats_missing}"
            
            return goal_reached, full_feedback
            
        except Exception as e:
            print(f"⚠️ LLM goal judge failed: {e}")
            return False, f"Judge error: {e}"
    
    def _rule_based_goal_judge(self, command: str, current_positions: Dict[str, float],
                               current_square: Optional[str]) -> Tuple[bool, str]:
        """Rule-based goal completion judge (fallback when no vision)."""
        command_lower = command.lower()
        
        # Check for square-based goals
        if "a1" in command_lower or "square a1" in command_lower:
            if current_square == "a1":
                return True, "Gripper is over square a1"
            else:
                return False, f"Not at a1 (currently at {current_square if current_square else 'unknown square'})"
        
        # Default: be conservative without vision
        return False, "Goal status unknown without vision; continue iterations."
    
    def _load_saved_positions(self) -> dict:
        """Load saved positions from file for LLM context."""
        positions_file = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/saved_positions.json"
        if positions_file.exists():
            try:
                with open(positions_file) as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _build_llm_prompt(self, command: str, current_positions: Dict[str, float]) -> str:
        """Build prompt for LLM with robot context and position model."""
        # Update position model before building prompt
        self.update_position_model()
        
        # Get current end-effector position
        ee_pos = self.position_model.get("end_effector", (0, 0, 0))
        ee_x, ee_y, ee_z = ee_pos
        
        # Get current square if calibration available
        current_square = None
        if "shoulder_pan" in current_positions and "shoulder_lift" in current_positions:
            current_square = self.calculate_robot_square(
                current_positions["shoulder_pan"],
                current_positions["shoulder_lift"]
            )
        
        # Get chess board calibration info
        board_info = self.get_chess_board_info()
        board_context = ""
        if board_info:
            board_context = f"""
Chess Board Calibration:
- Board is calibrated and ready for square-based movements
- Reference square: {board_info.get('reference_square', 'unknown')}
- Degrees per file (a->h): {board_info.get('degrees_per_file', 0):.2f}°
- Degrees per rank (1->8): {board_info.get('degrees_per_rank', 0):.2f}°
- Current square: {current_square if current_square else "unknown"}
- To move to a square (e.g., "a1", "e4"), calculate:
  * shoulder_pan = reference_pan + (file_index * degrees_per_file)
  * shoulder_lift = reference_lift + (rank_index * degrees_per_rank)
  * file_index: a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7
  * rank_index: 1=0, 2=1, 3=2, 4=3, 5=4, 6=5, 7=6, 8=7
"""

        # Workspace estimate (from mechanical/calibration limits)
        workspace_context = ""
        if getattr(self, "workspace_estimate", None):
            we = self.workspace_estimate
            b = (we or {}).get("ee_percentiles_mm") or {}
            jl = (we or {}).get("joint_limits_deg") or {}
            rec = (we or {}).get("recommended_scan_pose_deg") or {}
            workspace_context = f"""
WORKSPACE ESTIMATE (from your mechanical/calibration limits; use this to avoid 'looking the wrong way'):
- Reachable EE p5..p95 (mm): x={b.get('x_p5', 0):.0f}..{b.get('x_p95', 0):.0f}, y={b.get('y_p5', 0):.0f}..{b.get('y_p95', 0):.0f}, z={b.get('z_p5', 0):.0f}..{b.get('z_p95', 0):.0f}
- Joint limits (deg): {json.dumps(jl)}
- Recommended SCAN pose (deg): {json.dumps(rec)}
Notes:
- Keep wrist_roll near 0° during scanning (prevents camera from looking sideways).
- Keep wrist_flex negative (e.g. -20° to -45°) to look down at the board.
"""

        # Board-in-view localization (camera + motor 1)
        board_view_context = ""
        try:
            bvc = getattr(self, "board_view_calibration", None) or {}
            pan_center = bvc.get("pan_center_deg", None)
            if pan_center is not None and "shoulder_pan" in current_positions:
                m1_err = float(current_positions.get("shoulder_pan", 0.0)) - float(pan_center)
                board_view_context = f"""
BOARD VIEW LOCALIZATION (camera + motor feedback):
- Learned board-centered shoulder_pan (m1): pan_center_deg={float(pan_center):+.1f}°
- Current m1 centeredness error: (shoulder_pan - pan_center_deg) = {m1_err:+.1f}°
- Use shoulder_pan as the PRIMARY left/right centering control while scanning/approaching.
- Keep wrist_roll near 0° during scanning (prevents looking sideways).
"""
        except Exception:
            board_view_context = ""
        
        # Load saved/calibrated positions for reference
        saved_positions = self._load_saved_positions()
        saved_positions_context = ""
        if saved_positions:
            saved_positions_context = "\n\nSAVED/CALIBRATED POSITIONS (use these as reference waypoints):\n"
            for name, data in saved_positions.items():
                desc = data.get("description", "")
                positions = data.get("positions", {})
                pos_str = ", ".join([f"{k}:{v:.1f}°" if k != "gripper" else f"{k}:{v:.1f}%" for k, v in positions.items()])
                saved_positions_context += f"- '{name}': {desc}\n  Joint values: {pos_str}\n"
            saved_positions_context += """
You can use these saved positions to:
1. GO TO a named position directly (e.g., "go to home", "move to overview")
2. INTERPOLATE between positions for smooth movement
3. Use as WAYPOINTS in multi-step sequences
4. Reference them when calculating new positions
"""
        
        prompt = f"""You control a 5-DOF robot arm for chess piece manipulation.

⚠️ CRITICAL: THE CAMERA IS MOUNTED ON THE GRIPPER
- The camera sees what the gripper sees
- To "zoom out" and see the full board: RAISE the arm (lift shoulder, extend elbow)
- To "zoom in" on a square: LOWER the arm toward the board
- When close to board, camera sees only a small area around gripper

MOTOR REFERENCE (what each motor does):
- shoulder_pan: Rotates arm LEFT/RIGHT (negative=left, positive=right)
- shoulder_lift: Tilts arm UP/DOWN (negative=down toward board, positive=up away)
- elbow_flex: Extends/retracts forearm (negative=retracted/close, positive=extended/far)
- wrist_flex: Angles gripper UP/DOWN (negative=gripper points down, positive=gripper points up)
- gripper: Opens/closes jaws (0=fully open, 100=fully closed)

KEY POSITIONS (approximate joint angles):
┌─────────────────────────────────────────────────────────────────────┐
│ REST/HOME POSITION (arm folded, safe):                              │
│   shoulder_pan: 0°, shoulder_lift: 10°, elbow_flex: 20°,           │
│   wrist_flex: 0°, gripper: 0                                        │
├─────────────────────────────────────────────────────────────────────┤
│ OVERVIEW POSITION (zoomed out, sees full board):                    │
│   shoulder_pan: 0°, shoulder_lift: 30° to 50°, elbow_flex: 40°,    │
│   wrist_flex: -30° (gripper points down at board)                   │
├─────────────────────────────────────────────────────────────────────┤
│ CHESS WORKING HEIGHT (above board, ready to move):                  │
│   shoulder_lift: -30° to -50°, elbow_flex: -30° to -50°,           │
│   wrist_flex: 0° to 10° (gripper nearly vertical)                   │
├─────────────────────────────────────────────────────────────────────┤
│ TOUCHING BOARD (gripper at board surface):                          │
│   shoulder_lift: -60° to -80°, elbow_flex: -50° to -70°,           │
│   wrist_flex: -5° to 5° (gripper vertical)                          │
└─────────────────────────────────────────────────────────────────────┘

CURRENT ROBOT STATE:
- Joint positions: {json.dumps(current_positions, indent=2)}
- End-effector position: x={ee_x:.1f}mm, y={ee_y:.1f}mm, z={ee_z:.1f}mm
- Gripper state: {self.position_model.get('gripper', 0):.1f}%
{board_context}{workspace_context}{board_view_context}{saved_positions_context}
USER COMMAND: {command}

CAPABILITIES:
1. GRIPPER-MOUNTED CAMERA VISION:
   - Camera moves WITH the gripper
   - To see full board: GO TO OVERVIEW POSITION FIRST (shoulder_lift ~40°, elbow_flex ~40°)
   - Then analyze what you see before moving closer
   - To see a specific square up close: lower arm toward that square

2. SEQUENCE PLANNING (REQUIRED for find/locate tasks):
   - Step 1: ALWAYS start by moving to OVERVIEW position to see the full board
   - Step 2: From overview, identify target location visually
   - Step 3: Move arm toward target incrementally (pan + lift)
   - Step 4: Lower arm to get closer to target
   - Step 5: Fine-tune position and touch if needed

3. CHESS BOARD NAVIGATION:
   - shoulder_pan controls which FILE (a-h, left-right)
   - shoulder_lift + elbow_flex control which RANK (1-8, near-far) and HEIGHT
   - a1 is typically at negative pan angles (left side)
   - h8 is typically at positive pan angles (right side)

OUTPUT FORMAT:
For sequences (REQUIRED for "find X" or "go to X" tasks):
{{
  "sequence": [
    {{
      "step": 1,
      "action": {{"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 40.0, "elbow_flex.pos": 40.0, "wrist_flex.pos": -30.0}},
      "description": "Move to OVERVIEW position - raise arm to see full board",
      "wait_after": 0.8
    }},
    {{
      "step": 2,
      "action": {{"shoulder_pan.pos": -20.0, "shoulder_lift.pos": 20.0}},
      "description": "Pan toward left side of board where a-file is located",
      "wait_after": 0.5
    }},
    {{
      "step": 3,
      "action": {{"shoulder_lift.pos": -20.0, "elbow_flex.pos": -20.0}},
      "description": "Lower arm toward board to get closer to target",
      "wait_after": 0.5
    }},
    {{
      "step": 4,
      "action": {{"shoulder_lift.pos": -50.0, "elbow_flex.pos": -50.0, "wrist_flex.pos": 0.0}},
      "description": "Final approach - lower gripper to touch target square",
      "wait_after": 0.5
    }}
  ],
  "explanation": "Plan: 1) Raise to overview to see board, 2) Pan toward a-file, 3) Lower incrementally, 4) Touch target"
}}

For single actions (simple movements only):
{{
  "action": {{"shoulder_pan.pos": <float>, ...}},
  "explanation": "<brief explanation>"
}}

CRITICAL RULES:
1. For ANY "find", "locate", "go to" task: ALWAYS start with OVERVIEW position
2. Camera is on gripper - you can't see the board when arm is folded at rest
3. Move to overview FIRST (shoulder_lift ~40°, elbow_flex ~40°, wrist_flex ~-30°)
4. Then pan/tilt to find target visually
5. Then lower incrementally to approach
6. You can make larger movements (up to 40° per motor per step) - system will handle overload detection
7. Use wait_after 0.5-1.0s for movements
8. You have up to 25 iterations to reach the goal - be methodical and make progress each step
9. A fresh camera image is captured before each iteration - use it to verify progress
"""
        return prompt
    
    def _validate_llm_action(self, action: Dict[str, Any], current_positions: Dict[str, float], 
                              max_change: float = 30.0, is_sequence_step: bool = False) -> Optional[Dict[str, float]]:
        """Validate and clamp LLM-generated action to safe values.
        
        Args:
            action: Dict of motor positions
            current_positions: Current/expected positions to validate against
            max_change: Maximum degrees change per motor (default 30°)
            is_sequence_step: If True, allow slightly larger changes since sequence is pre-planned
        """
        if not isinstance(action, dict):
            print("⚠️ Validation failed: action is not a dict")
            return None
        
        safe_action = {}
        
        # More permissive movement limits - rely on overload detection for real safety
        if is_sequence_step:
            max_change = 40.0  # Larger for planned sequences
        else:
            max_change = 30.0  # More permissive for single actions

        # Joint-specific bounds (degrees): prefer measured/calibration-derived limits if available.
        joint_bounds_deg = {
            "shoulder_pan": (-110.0, 110.0),
            "shoulder_lift": (-100.0, 100.0),
            "elbow_flex": (-100.0, 100.0),
            "wrist_flex": (-100.0, 100.0),
            "wrist_roll": (-170.0, 170.0),
        }
        try:
            we = getattr(self, "workspace_estimate", None)
            jl = (we or {}).get("joint_limits_deg") if we else None
            if isinstance(jl, dict):
                for j, lim in jl.items():
                    if j in joint_bounds_deg and isinstance(lim, dict) and "min" in lim and "max" in lim:
                        joint_bounds_deg[j] = (float(lim["min"]), float(lim["max"]))
        except Exception:
            pass
        
        for motor_name, value in action.items():
            # Remove .pos suffix if present
            motor_key = motor_name.replace(".pos", "")
            
            if motor_key not in self.all_motors:
                print(f"⚠️ Unknown motor: {motor_key}")
                continue
            
            try:
                target_value = float(value)
                current_value = current_positions.get(motor_key, 0)
                
                # Check change magnitude
                change = abs(target_value - current_value)
                if change > max_change:
                    # Log the clamping
                    print(f"⚠️ Clamping {motor_key}: {current_value:.1f}° → {target_value:.1f}° (Δ{change:.1f}° > {max_change}°)")
                    # Clamp to max change
                    if target_value > current_value:
                        target_value = current_value + max_change
                    else:
                        target_value = current_value - max_change
                
                # Gripper-specific validation
                if motor_key == "gripper":
                    target_value = max(0.0, min(100.0, target_value))
                else:
                    # Joint angle validation (tighter, motor-specific bounds)
                    lo, hi = joint_bounds_deg.get(motor_key, (-180.0, 180.0))
                    target_value = max(lo, min(hi, target_value))
                
                safe_action[f"{motor_key}.pos"] = target_value
                
            except (ValueError, TypeError) as e:
                print(f"⚠️ Invalid value for {motor_key}: {value} ({e})")
                continue
        
        if not safe_action:
            print("⚠️ Validation failed: no valid motors in action")

        if not safe_action:
            return None

        # Extra safety: if kinematics is available, limit how much the LLM can DROP the end-effector Z per step.
        # This reduces repeated "banging down" even when the LLM keeps requesting to lower.
        try:
            if self.kinematics is not None:
                cur_j = {k: v for k, v in current_positions.items() if k != "gripper"}
                _, _, z_cur, _ = self.calculate_base_coordinates(cur_j)

                pred = dict(current_positions)
                for k, v in safe_action.items():
                    pred[k.replace(".pos", "")] = float(v)
                pred_j = {k: v for k, v in pred.items() if k != "gripper"}
                _, _, z_pred, _ = self.calculate_base_coordinates(pred_j)

                if isinstance(z_cur, (int, float)) and isinstance(z_pred, (int, float)):
                    max_z_drop_mm = 20.0 if is_sequence_step else 15.0
                    if (z_cur - z_pred) > max_z_drop_mm:
                        lo, hi = 0.0, 1.0
                        best = 0.0
                        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
                        base_targets = {j: float(safe_action.get(f"{j}.pos", current_positions.get(j, 0.0))) for j in joints}
                        base_curr = {j: float(current_positions.get(j, 0.0)) for j in joints}

                        for _ in range(10):
                            mid = (lo + hi) / 2.0
                            trial = dict(current_positions)
                            for j in joints:
                                trial[j] = base_curr[j] + (base_targets[j] - base_curr[j]) * mid
                            trial_j = {k: v for k, v in trial.items() if k != "gripper"}
                            _, _, z_trial, _ = self.calculate_base_coordinates(trial_j)
                            if isinstance(z_trial, (int, float)) and (z_cur - z_trial) <= max_z_drop_mm:
                                best = mid
                                lo = mid
                            else:
                                hi = mid

                        if best < 1.0:
                            for j in joints:
                                if f"{j}.pos" in safe_action:
                                    safe_action[f"{j}.pos"] = base_curr[j] + (base_targets[j] - base_curr[j]) * best
                            print(f"⚠️ Z-drop safety: scaled action to {best:.2f} to limit Δz to {max_z_drop_mm:.0f}mm")
        except Exception:
            pass
        
        return safe_action
    
    def _execute_llm_action(self, action: Dict[str, float]):
        """Execute validated LLM action on robot with COORDINATED multi-motor movement.
        
        Uses sync_write to move all motors SIMULTANEOUSLY, avoiding interference
        when motors need to move together (like elbow needing shoulder to adjust).
        """
        try:
            # Build dict of motor positions for sync_write
            motor_positions = {}
            for motor_key, value in action.items():
                motor_name = motor_key.replace(".pos", "")
                if motor_name in self.all_motors:
                    motor_positions[motor_name] = value
            
            if not motor_positions:
                print("   ⚠️ No valid motor positions in action")
                return
            
            # Log what we're about to do
            print(f"   🔄 Coordinated move: {', '.join([f'{k}={v:.1f}°' for k,v in motor_positions.items()])}")
            
            try:
                # Use sync_write to move ALL motors SIMULTANEOUSLY
                # This avoids interference issues where one motor blocks another
                self.bus.sync_write("Goal_Position", motor_positions, normalize=True)
                print(f"   ✓ sync_write completed for {len(motor_positions)} motors")
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"   ⚠️ sync_write failed: {e}")
                
                # Check for specific errors
                if "overload" in error_msg:
                    print("   🔥 OVERLOAD detected!")
                    print("   ⏸️ Pausing 3 seconds due to overload...")
                    time.sleep(3.0)
                    raise RuntimeError("Motor overload detected - action stopped for safety")
                
                if "port is in use" in error_msg or ("port" in error_msg and "use" in error_msg):
                    print("   🔌 PORT CONFLICT detected")
                    print("   ⏸️ Pausing 1 second...")
                    time.sleep(1.0)
                
                # Fall back to sequential writes if sync_write fails
                print("   📝 Falling back to sequential writes...")
                for motor_name, value in motor_positions.items():
                    try:
                        self.bus.write("Goal_Position", motor_name, value, normalize=True)
                        time.sleep(0.05)  # Minimal delay between motors
                    except Exception as e2:
                        print(f"   ⚠️ {motor_name} write error: {e2}")
            
            # Wait for motors to reach target
            time.sleep(0.5)
            
            # Update position model after execution
            self.update_position_model()
            
            self.status_bar.setText("✅ LLM action executed")
            
        except RuntimeError:
            raise  # Re-raise overload errors
        except Exception as e:
            self.status_bar.setText(f"❌ Execution error: {str(e)[:50]}...")
            raise
    
    def _execute_llm_sequence(self, sequence: List[Dict], initial_positions: Dict[str, float], original_command: str = "") -> bool:
        """Execute a sequence of LLM actions with ADAPTIVE vision feedback loop.
        
        The feedback loop actually adjusts subsequent steps based on what the LLM sees:
        1. Execute step
        2. Capture image and analyze with LLM
        3. LLM decides if adjustment needed
        4. If yes, LLM generates corrective action that replaces/modifies next steps
        """
        try:
            # Track both actual positions (from robot) and expected positions (from plan)
            actual_positions = initial_positions.copy()
            expected_positions = initial_positions.copy()  # Track what we EXPECT after each step
            executed_steps = 0
            remaining_sequence = list(sequence)  # Mutable copy we can modify
            
            print(f"🚀 Starting sequence with {len(sequence)} steps")
            print(f"   Initial positions: {json.dumps({k: f'{v:.1f}°' for k, v in initial_positions.items()})}")
            
            while remaining_sequence:
                step_data = remaining_sequence.pop(0)
                step_num = step_data.get("step", executed_steps + 1)
                step_action = step_data.get("action", {})
                step_desc = step_data.get("description", f"Step {step_num}")
                wait_after = step_data.get("wait_after", 0.5)
                
                print(f"\n📍 Step {step_num}: {step_desc}")
                
                # Try to update actual positions from robot, but don't fail if it errors
                try:
                    self.update_position_model()
                    for motor_name in self.all_motors.keys():
                        if motor_name in self.position_model["joints"]:
                            actual_positions[motor_name] = self.position_model["joints"][motor_name]
                except Exception as e:
                    print(f"   ⚠️ Position read error: {e}, using expected positions")
                
                # Use EXPECTED positions for validation (what we planned to reach)
                # This is more reliable than actual positions which may lag or have read errors
                validation_positions = expected_positions.copy()
                
                # Validate this step with sequence-aware settings (more permissive)
                safe_action = self._validate_llm_action(step_action, validation_positions, is_sequence_step=True)
                
                if not safe_action:
                    self.status_bar.setText(f"⚠️ Step {step_num} failed validation: {step_desc}")
                    print(f"⚠️ Skipping step {step_num}: validation failed")
                    continue
                
                # Execute the step with COORDINATED multi-motor movement
                try:
                    self.status_bar.setText(f"🔄 Executing step {step_num}/{len(sequence)}: {step_desc}")
                    print(f"   Sending commands: {json.dumps({k: f'{v:.1f}°' for k, v in safe_action.items()})}")
                    
                    # Build dict of motor positions for sync_write
                    motor_positions = {}
                    for motor_key, value in safe_action.items():
                        motor_name = motor_key.replace(".pos", "")
                        if motor_name in self.all_motors:
                            motor_positions[motor_name] = value
                    
                    # Track errors for overload/port conflict detection
                    overload_detected = False
                    port_conflict = False
                    
                    try:
                        # Use sync_write to move ALL motors SIMULTANEOUSLY
                        # This avoids interference issues where one motor blocks another
                        self.bus.sync_write("Goal_Position", motor_positions, normalize=True)
                        print(f"   ✓ sync_write: {len(motor_positions)} motors moved together")
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        print(f"   ⚠️ sync_write failed: {e}")
                        
                        # Check for specific errors
                        if "overload" in error_msg:
                            overload_detected = True
                            print(f"   🔥 OVERLOAD detected - stopping sequence")
                        
                        if "port is in use" in error_msg or ("port" in error_msg and "use" in error_msg):
                            port_conflict = True
                            print(f"   🔌 PORT CONFLICT detected - pausing")
                        
                        # Fall back to sequential writes if sync_write fails
                        if not overload_detected and not port_conflict:
                            print("   📝 Falling back to sequential writes...")
                            for motor_name, value in motor_positions.items():
                                try:
                                    self.bus.write("Goal_Position", motor_name, value, normalize=True)
                                    time.sleep(0.05)  # Minimal delay
                                except Exception as e2:
                                    print(f"   ⚠️ {motor_name} write error: {e2}")
                    
                    # If overload or port conflict, stop sequence and wait
                    if overload_detected:
                        self.status_bar.setText(f"⚠️ Motor overload detected - stopping sequence for safety")
                        print("   🔥 OVERLOAD detected - stopping sequence and pausing 5 seconds...")
                        time.sleep(5.0)  # Critical - longer cool-down for motor safety
                        # Don't continue sequence - return False to stop
                        return False
                    
                    if port_conflict:
                        self.status_bar.setText(f"⚠️ Port conflict - pausing 1 second...")
                        print("   ⏸️ Pausing 1 second due to port conflict...")
                        time.sleep(1.0)  # Brief pause for port to free up
                        # Continue but with longer wait
                        wait_after = max(wait_after, 1.5)
                    
                    # Wait for motors to reach position
                    time.sleep(wait_after)
                    
                    # Update EXPECTED positions with what we commanded (for next step validation)
                    for motor_key, value in safe_action.items():
                        motor_name = motor_key.replace(".pos", "")
                        expected_positions[motor_name] = value
                    
                    # Also try to update actual positions from robot
                    try:
                        self.update_position_model()
                        for motor_name in self.all_motors.keys():
                            if motor_name in self.position_model["joints"]:
                                actual_positions[motor_name] = self.position_model["joints"][motor_name]
                    except Exception as e:
                        print(f"   ⚠️ Position read error: {e}")
                    
                    executed_steps += 1
                    print(f"✅ Step {step_num} completed: {step_desc}")
                    
                    # ADAPTIVE VISION FEEDBACK LOOP
                    # This actually adjusts subsequent steps based on what the LLM sees
                    if self.vision_enabled_checkbox.isChecked() and self.llm_enabled and remaining_sequence:
                        try:
                            camera_source = self.vision_camera_combo.currentText().split()[0].lower()
                            image_data = self.capture_camera_image(camera_source)
                            
                            if image_data:
                                self.status_bar.setText(f"👁️ Analyzing scene after step {step_num}...")
                                
                                # Get adaptive feedback - LLM can suggest adjustments
                                adjustment = self._get_adaptive_feedback(
                                    original_command=original_command,
                                    completed_step=step_desc,
                                    remaining_steps=[s.get("description", "") for s in remaining_sequence],
                                    current_positions=expected_positions,  # Use expected positions for context
                                    image_data=image_data
                                )
                                
                                if adjustment:
                                    if adjustment.get("needs_adjustment", False):
                                        # LLM wants to modify the plan
                                        new_steps = adjustment.get("corrective_steps", [])
                                        feedback_text = adjustment.get("feedback", "Adjusting plan...")
                                        
                                        print(f"🔄 Vision feedback: {feedback_text}")
                                        print(f"   Inserting {len(new_steps)} corrective steps")
                                        
                                        # Display feedback
                                        current_reasoning = self.llm_reasoning_display.toPlainText()
                                        self.llm_reasoning_display.setText(
                                            current_reasoning + f"\n\n👁️ ADAPTIVE ADJUSTMENT after step {step_num}:\n{feedback_text}\n→ Adding {len(new_steps)} corrective steps"
                                        )
                                        
                                        # INSERT corrective steps at the front of remaining sequence
                                        for i, new_step in enumerate(reversed(new_steps)):
                                            new_step["step"] = f"{step_num}.{len(new_steps)-i}"  # e.g., "3.1", "3.2"
                                            remaining_sequence.insert(0, new_step)
                                    else:
                                        # LLM says we're on track
                                        feedback_text = adjustment.get("feedback", "On track")
                                        print(f"👁️ Vision feedback: {feedback_text}")
                                        current_reasoning = self.llm_reasoning_display.toPlainText()
                                        self.llm_reasoning_display.setText(
                                            current_reasoning + f"\n\n👁️ After step {step_num}: {feedback_text}"
                                        )
                        except Exception as e:
                            print(f"⚠️ Adaptive feedback error: {e}")
                    
                except Exception as e:
                    self.status_bar.setText(f"❌ Step {step_num} execution error: {str(e)[:50]}...")
                    print(f"❌ Step {step_num} failed: {e}")
                    continue
            
            return executed_steps > 0
            
        except Exception as e:
            self.status_bar.setText(f"❌ Sequence execution error: {str(e)[:50]}...")
            print(f"❌ Sequence execution failed: {e}")
            return False
    
    def _get_adaptive_feedback(self, original_command: str, completed_step: str, 
                                remaining_steps: List[str], current_positions: Dict[str, float],
                                image_data) -> Optional[Dict]:
        """Get adaptive feedback from LLM - can suggest corrective steps.
        
        Returns:
            {
                "needs_adjustment": bool,
                "feedback": str,
                "corrective_steps": [{"action": {...}, "description": "...", "wait_after": 0.5}, ...]
            }
        """
        try:
            if not self.llm_enabled or not self.llm_client:
                return None
            
            selected_model = self.llm_model_combo.currentText().replace(" 👁️", "").strip()
            
            # Get current square if available
            current_square = None
            if "shoulder_pan" in current_positions and "shoulder_lift" in current_positions:
                current_square = self.calculate_robot_square(
                    current_positions["shoulder_pan"],
                    current_positions["shoulder_lift"]
                )
            
            # Build adaptive feedback prompt
            prompt = f"""ADAPTIVE FEEDBACK REQUEST

Original task: {original_command}
Just completed: {completed_step}
Remaining planned steps: {remaining_steps if remaining_steps else "None"}

Current robot state:
- Joint positions: {json.dumps(current_positions, indent=2)}
- Current square (if on board): {current_square if current_square else "unknown"}

Analyze the image and decide if adjustment is needed.

Output JSON:
{{
  "needs_adjustment": <true if plan needs correction, false if on track>,
  "feedback": "<brief description of what you see and whether target is visible/reachable>",
  "corrective_steps": [
    // ONLY if needs_adjustment is true, provide 1-3 corrective steps:
    {{
      "action": {{"shoulder_pan.pos": <float>, "shoulder_lift.pos": <float>, ...}},
      "description": "<what this corrective step does>",
      "wait_after": 0.3
    }}
  ]
}}

Guidelines:
- If target is visible and we're on track, set needs_adjustment: false
- If target is not visible or we're off course, suggest corrective steps
- Keep corrective steps small (max 10-15 degrees per motor)
- Focus on getting the target in view or getting closer to it
"""
            
            # Build vision message
            user_content = [{"type": "text", "text": prompt}]
            
            if isinstance(image_data, list):
                for img in image_data:
                    if img:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "low"}
                        })
            else:
                if image_data:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "low"}
                    })
            
            messages = [
                {"role": "system", "content": "You are a robot control assistant with vision. Analyze scenes and suggest adjustments. Output only valid JSON."},
                {"role": "user", "content": user_content}
            ]
            
            # Get response - use model-appropriate parameters
            api_params = {
                "model": selected_model,
                "messages": messages,
            }
            if not selected_model.startswith("o"):
                api_params["max_tokens"] = 500
                api_params["response_format"] = {"type": "json_object"}
            else:
                api_params["max_completion_tokens"] = 500
            response = self.llm_client.chat.completions.create(**api_params)
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"⚠️ Adaptive feedback failed: {e}")
            return None
    
    def _move_gripper_mm(self, dx_mm, dy_mm, dz_mm, apply_ui_step: bool = True):
        """Move gripper in base coordinates by the specified delta (in mm).

        If apply_ui_step=True, the UI step-size multiplier is applied (for button nudges).
        If apply_ui_step=False, dx_mm/dy_mm/dz_mm are treated as direct millimeters (for autopick logic).
        """
        try:
            if apply_ui_step:
                step_size = float(self.step_size_combo.currentText())
                dx = float(dx_mm) * step_size / 10.0
                dy = float(dy_mm) * step_size / 10.0
                dz = float(dz_mm) * step_size / 10.0
            else:
                dx = float(dx_mm)
                dy = float(dy_mm)
                dz = float(dz_mm)

            # Safety: avoid "slamming" downward into the board/table.
            max_down_mm = 8.0
            if dz < -max_down_mm:
                dz = -max_down_mm
            
            self.status_bar.setText(f"🎯 Moving gripper: Δx={dx:.1f}, Δy={dy:.1f}, Δz={dz:.1f} mm")
            
            # Get current joint positions
            current_joints = {}
            for motor_name in self.all_motors.keys():
                if motor_name != "gripper":
                    # Prefer live read, but fall back to last-known to avoid hard failures mid-run.
                    last_known = None
                    try:
                        last_known = self.position_model.get("joints", {}).get(motor_name, None)
                    except Exception:
                        last_known = None
                    try:
                        pos = self.bus.read("Present_Position", motor_name, normalize=True, num_retry=2)
                        current_joints[motor_name] = pos
                    except Exception as e:
                        self._append_exec_log("error", f"Movement read failed: Present_Position {motor_name}: {e}")
                        if last_known is not None:
                            current_joints[motor_name] = float(last_known)
                            self._append_exec_log("warning", f"Using last-known {motor_name}={float(last_known):.1f}° for IK")
                        else:
                            raise
            
            # Use inverse kinematics if available
            if self.kinematics is not None:
                try:
                    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
                    current_joint_array = [float(current_joints[name]) for name in joint_names]

                    # Preserve current orientation; translate only.
                    T_current = self.kinematics.forward_kinematics(current_joint_array)
                    T_target = np.array(T_current, copy=True)
                    T_target[0, 3] += dx / 1000.0
                    T_target[1, 3] += dy / 1000.0
                    T_target[2, 3] += dz / 1000.0

                    # Check workspace bounds in mm (coarse safety)
                    target_x = float(T_target[0, 3] * 1000.0)
                    target_y = float(T_target[1, 3] * 1000.0)
                    target_z = float(T_target[2, 3] * 1000.0)
                    distance = float(np.sqrt(target_x**2 + target_y**2 + target_z**2))
                    if distance > 450:
                        self.status_bar.setText("❌ Target outside workspace bounds!")
                        return
                    if distance < 80:
                        self.status_bar.setText("❌ Target too close to base!")
                        return

                    # Solve IK
                    target_joints = self.kinematics.inverse_kinematics(
                        current_joint_array, T_target, position_weight=1.0, orientation_weight=0.02
                    )

                    # Clamp per-joint change (deg) and move all joints together
                    max_joint_step_deg = 8.0
                    motor_positions = {}
                    for i, motor_name in enumerate(joint_names):
                        cur = float(current_joints[motor_name])
                        tgt = float(target_joints[i]) if i < len(target_joints) else cur
                        delta = tgt - cur
                        if abs(delta) > max_joint_step_deg:
                            tgt = cur + max_joint_step_deg * (1.0 if delta > 0 else -1.0)
                        motor_positions[motor_name] = tgt

                    self.bus.sync_write("Goal_Position", motor_positions, normalize=True)
                    time.sleep(0.4)

                    self.status_bar.setText(f"✅ Moved toward ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
                    
                except Exception as e:
                    self.status_bar.setText(f"❌ IK failed: {str(e)[:50]}...")
            else:
                # Simple joint-space movement approximation
                self.status_bar.setText("⚠️ Simple movement (no IK available)")
                
                # Small movements in joint space
                if abs(dx) > abs(dy):
                    if dx > 0:
                        self.bus.write("Goal_Position", "elbow_flex", current_joints["elbow_flex"] + 2)
                    else:
                        self.bus.write("Goal_Position", "elbow_flex", current_joints["elbow_flex"] - 2)
                else:
                    if dy > 0:
                        self.bus.write("Goal_Position", "shoulder_pan", current_joints["shoulder_pan"] + 2)
                    else:
                        self.bus.write("Goal_Position", "shoulder_pan", current_joints["shoulder_pan"] - 2)
                
                time.sleep(1)
                self.status_bar.setText("✅ Joint movement completed")
        
        except Exception as e:
            # Full error goes to execution log; status bar is short by design.
            self._append_exec_log("error", f"Movement failed: {e}")
            self.status_bar.setText("❌ Movement failed (see Execution Log)")

    def move_gripper(self, dx_mm, dy_mm, dz_mm):
        """UI-friendly wrapper: uses the step-size combo scaling."""
        return self._move_gripper_mm(dx_mm, dy_mm, dz_mm, apply_ui_step=True)
    
    def update_workspace_position(self, x_mm, y_mm, z_mm):
        """Update robot position on workspace visualization."""
        self.workspace_canvas.set_robot_position(x_mm, y_mm, z_mm)
    
    def update_workspace_display(self, x_mm, y_mm, z_mm):
        """Update workspace visualization with current robot position."""
        self.workspace_side_canvas.set_robot_position(x_mm, y_mm, z_mm)
        self.workspace_top_canvas.set_robot_position(x_mm, y_mm, z_mm)
    
    def update_position_model(self):
        """Update internal position model with current robot state."""
        try:
            # Read current joint positions
            for motor_name in self.all_motors.keys():
                try:
                    pos = self.bus.read("Present_Position", motor_name, normalize=True)
                    self.position_model["joints"][motor_name] = pos
                except:
                    pass
            
            # Calculate end-effector position
            joint_positions = {k: v for k, v in self.position_model["joints"].items() if k != "gripper"}
            x, y, z, method = self.calculate_base_coordinates(joint_positions)
            self.position_model["end_effector"] = (x, y, z)
            
            # Get gripper state
            self.position_model["gripper"] = self.position_model["joints"].get("gripper", 0.0)
            
            # Update timestamp
            self.position_model["last_update"] = time.time()
            
        except Exception as e:
            print(f"⚠️ Position model update error: {e}")
    
    def calculate_base_coordinates(self, joint_positions):
        """Calculate end-effector position in robot base frame."""
        try:
            if self.kinematics is not None:
                # Use full forward kinematics
                joint_angles = [joint_positions[name] for name in 
                              ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]]
                
                T_base_ee = self.kinematics.forward_kinematics(joint_angles)
                
                # Extract position from transformation matrix
                x_mm = T_base_ee[0, 3] * 1000
                y_mm = T_base_ee[1, 3] * 1000
                z_mm = T_base_ee[2, 3] * 1000
                
                return x_mm, y_mm, z_mm, "Full FK"
                
            else:
                # Simplified approximation using joint angles
                shoulder_pan = np.radians(joint_positions["shoulder_pan"])
                shoulder_lift = np.radians(joint_positions["shoulder_lift"]) 
                elbow_flex = np.radians(joint_positions["elbow_flex"])
                wrist_flex = np.radians(joint_positions["wrist_flex"])
                
                # Approximate arm segment lengths (mm)
                L1 = 150
                L2 = 120
                L3 = 80
                
                # Forward kinematics approximation
                total_lift = shoulder_lift + elbow_flex
                
                # Calculate reach
                reach = L1 * np.cos(shoulder_lift) + L2 * np.cos(total_lift) + L3 * np.cos(total_lift + wrist_flex)
                height = L1 * np.sin(shoulder_lift) + L2 * np.sin(total_lift) + L3 * np.sin(total_lift + wrist_flex)
                
                # Apply shoulder pan rotation
                x_mm = reach * np.cos(shoulder_pan)
                y_mm = reach * np.sin(shoulder_pan)
                z_mm = height + 200
                
                return x_mm, y_mm, z_mm, "Approximation"
            
        except Exception as e:
            return 0, 0, 0, f"Error: {e}"
    
    def calculate_robot_square(self, shoulder_pan, shoulder_lift):
        """Calculate which chess square robot is over."""
        try:
            calib_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
            chess_calib_file = calib_dir / "chess_coordinate_system.json"
            
            if chess_calib_file.exists():
                with open(chess_calib_file) as f:
                    chess_system = json.load(f)
                
                if chess_system.get("coordinate_system_calculated", False):
                    ref_pos = chess_system["reference_position"]
                    degrees_per_file = chess_system["degrees_per_file"]
                    degrees_per_rank = chess_system["degrees_per_rank"]
                    
                    # Calculate file and rank
                    file_offset = (shoulder_pan - ref_pos["shoulder_pan"]) / degrees_per_file
                    rank_offset = (shoulder_lift - ref_pos["shoulder_lift"]) / degrees_per_rank
                    
                    file_idx = round(file_offset)
                    rank_idx = round(rank_offset)
                    
                    if 0 <= file_idx <= 7 and 0 <= rank_idx <= 7:
                        square = chr(ord('a') + file_idx) + str(rank_idx + 1)
                        return square
            
            return None
        except:
            return None
    
    def square_to_joint_positions(self, square: str) -> Optional[Dict[str, float]]:
        """Convert chess square (e.g., 'a1') to approximate joint positions.
        Returns dict with shoulder_pan and shoulder_lift, or None if calibration not available."""
        try:
            calib_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
            chess_calib_file = calib_dir / "chess_coordinate_system.json"
            
            if not chess_calib_file.exists():
                return None
            
            with open(chess_calib_file) as f:
                chess_system = json.load(f)
            
            if not chess_system.get("coordinate_system_calculated", False):
                return None
            
            # Parse square (e.g., "a1" -> file=0, rank=0)
            file_char = square[0].lower()
            rank_char = square[1]
            file_idx = ord(file_char) - ord('a')
            rank_idx = int(rank_char) - 1
            
            if not (0 <= file_idx <= 7 and 0 <= rank_idx <= 7):
                return None
            
            # Get calibration parameters
            ref_pos = chess_system["reference_position"]
            degrees_per_file = chess_system["degrees_per_file"]
            degrees_per_rank = chess_system["degrees_per_rank"]
            
            # Calculate target joint positions
            target_pan = ref_pos["shoulder_pan"] + (file_idx * degrees_per_file)
            target_lift = ref_pos["shoulder_lift"] + (rank_idx * degrees_per_rank)
            
            return {
                "shoulder_pan": target_pan,
                "shoulder_lift": target_lift
            }
        except Exception as e:
            print(f"⚠️ Square to position conversion error: {e}")
            return None
    
    def get_chess_board_info(self) -> Optional[Dict]:
        """Get chess board calibration information for LLM context."""
        try:
            calib_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
            chess_calib_file = calib_dir / "chess_coordinate_system.json"
            
            if not chess_calib_file.exists():
                return None
            
            with open(chess_calib_file) as f:
                chess_system = json.load(f)
            
            if not chess_system.get("coordinate_system_calculated", False):
                return None
            
            return {
                "calibrated": True,
                "reference_square": chess_system.get("reference_square", "unknown"),
                "degrees_per_file": chess_system.get("degrees_per_file", 0),
                "degrees_per_rank": chess_system.get("degrees_per_rank", 0),
                "reference_position": chess_system.get("reference_position", {})
            }
        except:
            return None
    
    def update_main_camera(self, pixmap):
        """Unused - kept for compatibility."""
        pass
    
    def update_gripper_camera(self, pixmap):
        """Update gripper camera view from monitoring thread."""
        self.gripper_camera_label.setPixmap(pixmap)
        self.gripper_camera_status.setText("[LIVE]")
        self.gripper_camera_status.setStyleSheet("color: #3fb950;")
        
        # Update FPS counter
        if not hasattr(self, '_gripper_frame_count'):
            self._gripper_frame_count = 0
            self._gripper_last_fps_time = time.time()
        self._gripper_frame_count += 1
        current_time = time.time()
        if current_time - self._gripper_last_fps_time >= 1.0:
            fps = self._gripper_frame_count / (current_time - self._gripper_last_fps_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._gripper_frame_count = 0
            self._gripper_last_fps_time = current_time
    
    def update_motors(self, motor_data):
        """Update motor positions and status from monitoring thread."""
        try:
            all_good = True
            
            for motor_name, data in motor_data.items():
                if data.get("status") == "ok" and data.get("position") is not None:
                    # Update all signal displays
                    signal_labels = self.motor_signal_labels.get(motor_name, {})
                    
                    # Position
                    pos = data.get("position")
                    if pos is not None and "position" in signal_labels:
                        unit = "%" if motor_name == "gripper" else "°"
                        signal_labels["position"].setText(f"{pos:6.1f}{unit}")
                    
                    # Goal Position
                    goal_pos = data.get("goal_position")
                    if goal_pos is not None and "goal_position" in signal_labels:
                        unit = "%" if motor_name == "gripper" else "°"
                        signal_labels["goal_position"].setText(f"{goal_pos:6.1f}{unit}")
                    elif "goal_position" in signal_labels:
                        signal_labels["goal_position"].setText("--")
                    
                    # Velocity
                    vel = data.get("velocity")
                    if vel is not None and "velocity" in signal_labels:
                        signal_labels["velocity"].setText(f"{vel:6.0f}")
                    elif "velocity" in signal_labels:
                        signal_labels["velocity"].setText("--")
                    
                    # Load (torque percentage)
                    load = data.get("load")
                    if load is not None and "load" in signal_labels:
                        # Load is typically in sign-magnitude format, convert to percentage
                        signal_labels["load"].setText(f"{load:6.1f}%")
                    elif "load" in signal_labels:
                        signal_labels["load"].setText("--")
                    
                    # Current
                    current = data.get("current")
                    if current is not None and "current" in signal_labels:
                        signal_labels["current"].setText(f"{current:6.0f}mA")
                    elif "current" in signal_labels:
                        signal_labels["current"].setText("--")
                    
                    # Voltage
                    voltage = data.get("voltage")
                    if voltage is not None and "voltage" in signal_labels:
                        # Voltage is typically in 0.1V units
                        signal_labels["voltage"].setText(f"{voltage/10.0:4.1f}V")
                    elif "voltage" in signal_labels:
                        signal_labels["voltage"].setText("--")
                    
                    # Temperature
                    temp = data.get("temperature")
                    if temp is not None and "temperature" in signal_labels:
                        signal_labels["temperature"].setText(f"{temp:3.0f}°C")
                    elif "temperature" in signal_labels:
                        signal_labels["temperature"].setText("--")
                    
                    # Moving status
                    moving = data.get("moving")
                    if moving is not None and "moving" in signal_labels:
                        signal_labels["moving"].setText("Yes" if moving else "No")
                    elif "moving" in signal_labels:
                        signal_labels["moving"].setText("--")
                    
                    # Torque Enable
                    torque = data.get("torque_enable")
                    if torque is not None and "torque_enable" in signal_labels:
                        signal_labels["torque_enable"].setText("ON" if torque else "OFF")
                    elif "torque_enable" in signal_labels:
                        signal_labels["torque_enable"].setText("--")
                    
                    # Update status indicator
                    self.motor_status_labels[motor_name].setText("[OK]")
                    self.motor_status_labels[motor_name].setStyleSheet("color: #3fb950;")
                else:
                    all_good = False
                    # Show error for all signals
                    signal_labels = self.motor_signal_labels.get(motor_name, {})
                    for label in signal_labels.values():
                        label.setText("ERROR")
                        label.setStyleSheet("""
                            font-size: 9pt;
                            font-weight: 600;
                            color: #e74c3c;
                            padding: 4px 8px;
                            background-color: #3d1e1e;
                            border-radius: 3px;
                            border: 1px solid #e74c3c;
                        """)
                    
                    self.motor_status_labels[motor_name].setText("[ERR]")
                    self.motor_status_labels[motor_name].setStyleSheet("color: #f85149;")
                
                # Update robot overall status
                if all_good:
                    self.robot_status.setText("[OK]")
                    self.robot_status.setStyleSheet("color: #3fb950; font-weight: bold;")
                    self.quick_status.setText("[READY]")
                    self.quick_status.setStyleSheet("color: #3fb950; font-weight: bold;")
                    
                    # Calculate robot base coordinates
                    joint_positions = {name: data["position"] for name, data in motor_data.items() 
                                     if data["position"] is not None}
                    
                    # Update 3D robot visualization
                    if hasattr(self, 'robot_3d_widget') and len(joint_positions) >= 5:
                        joint_angles = {
                            "shoulder_pan": joint_positions.get("shoulder_pan", 0),
                            "shoulder_lift": joint_positions.get("shoulder_lift", 0),
                            "elbow_flex": joint_positions.get("elbow_flex", 0),
                            "wrist_flex": joint_positions.get("wrist_flex", 0),
                            "wrist_roll": joint_positions.get("wrist_roll", 0),
                            "gripper": joint_positions.get("gripper", 0)
                        }
                        self.robot_3d_widget.update_joints(joint_angles)
                    
                    if len(joint_positions) >= 5:
                        # Calculate coordinates for internal use (not displayed)
                        x_mm, y_mm, z_mm, method = self.calculate_base_coordinates(joint_positions)
                        
                        # Update joint configuration display (if it exists)
                        pan = joint_positions.get("shoulder_pan", 0)
                        lift = joint_positions.get("shoulder_lift", 0)
                        elbow = joint_positions.get("elbow_flex", 0)
                        wrist_flex = joint_positions.get("wrist_flex", 0)
                        wrist_roll = joint_positions.get("wrist_roll", 0)
                        
                        # Only update if joint_config widget exists (from coordinates panel)
                        if hasattr(self, 'joint_config'):
                            config_text = f"Shoulder: {pan:.1f}°, {lift:.1f}° | Elbow: {elbow:.1f}° | Wrist: {wrist_flex:.1f}°, {wrist_roll:.1f}°"
                            self.joint_config.setText(config_text)
                        
                        # Chess board calculations removed - panel no longer displayed
                else:
                    self.robot_status.setText("[WARN]")
                    self.robot_status.setStyleSheet("color: #f0883e; font-weight: bold;")
                    self.quick_status.setText("[WARN]")
                    self.quick_status.setStyleSheet("color: #f0883e; font-weight: bold;")
                
        except Exception as e:
            self.robot_status.setText("[ERROR]")
            self.robot_status.setStyleSheet("color: #f85149; font-weight: bold;")
            self.quick_status.setText("[ERROR]")
            self.quick_status.setStyleSheet("color: #f85149; font-weight: bold;")
    
    def start_monitoring(self):
        """Start monitoring threads."""
        self.monitor_thread = MonitoringThread(self.bus, self.cameras)
        self.monitor_thread.main_camera_update.connect(self.update_main_camera)
        self.monitor_thread.gripper_camera_update.connect(self.update_gripper_camera)
        self.monitor_thread.motor_update.connect(self.update_motors)
        self.monitor_thread.status_update.connect(self.status_bar.setText)
        self.monitor_thread.running = True
        self.monitor_thread.start()
        
        self.status_bar.setText("🔄 Starting monitoring systems...")
    
    def stop_monitoring(self):
        """Stop monitoring and close application."""
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.stop()
            self.monitor_thread.wait()
        self.close()
    
    def go_home(self):
        """Move robot to home position."""
        try:
            home_path = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/home_joints.npy"
            
            if home_path.exists():
                home_positions = np.load(home_path)
                motor_names = list(self.all_motors.keys())
                
                self.status_bar.setText("🏠 Moving to home position...")
                
                # Move to home position
                for i, motor_name in enumerate(motor_names):
                    self.bus.write("Goal_Position", motor_name, float(home_positions[i]))
                    time.sleep(0.5)
                
                self.status_bar.setText("✅ Moved to home position")
            else:
                self.status_bar.setText("❌ No home position saved")
                
        except Exception as e:
            self.status_bar.setText(f"❌ Home move failed: {e}")
    
    def test_motors(self):
        """Test motors with coordinated movements to avoid interference."""
        if not hasattr(self, 'bus') or not self.bus.is_connected:
            self.status_bar.setText("❌ Robot not connected")
            return
        
        try:
            # Pause monitoring to avoid port conflicts
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                self.monitor_thread.paused = True
                time.sleep(0.2)
            
            print("🧪 Testing motors with COORDINATED movements...")
            print("   ℹ️ Moving multiple motors together to avoid interference")
            self.status_bar.setText("🧪 Testing motors...")
            
            # Coordinated movement amounts per step (small increments)
            # Moving all motors together avoids interference from arm structure
            step_movements = {
                "shoulder_pan": 1.0,       # 1 degree per step
                "shoulder_lift": 1.0,      # 1 degree per step  
                "elbow_flex": 1.0,         # 1 degree per step
                "wrist_flex": 1.0,         # 1 degree per step
                "wrist_roll": 2.0,         # 2 degrees per step (more rotation range)
                "gripper": 1.0             # 1% per step
            }
            
            num_steps = 5  # Total steps in each direction
            step_delay = 0.3  # Delay between steps (seconds)
            
            # Read current positions (normalized)
            print("\n📊 Reading current positions...")
            start_positions = {}
            for motor_name in self.all_motors.keys():
                try:
                    pos = self.bus.read("Present_Position", motor_name, normalize=True)
                    start_positions[motor_name] = pos
                    unit = "%" if motor_name == "gripper" else "°"
                    print(f"  {motor_name}: {pos:.1f}{unit}")
                except Exception as e:
                    print(f"  ⚠️ Could not read {motor_name}: {e}")
                    start_positions[motor_name] = None
            
            # Filter to only motors we can read
            working_motors = [m for m in self.all_motors.keys() if start_positions.get(m) is not None]
            print(f"\n✅ Working motors: {len(working_motors)}/{len(self.all_motors)}")
            
            # === PHASE 1: Move all motors together in positive direction ===
            print(f"\n🔄 PHASE 1: Moving all motors together (+{num_steps} steps)...")
            self.status_bar.setText("🧪 Phase 1: Coordinated positive movement...")
            
            current_positions = start_positions.copy()
            
            for step in range(num_steps):
                print(f"   Step {step + 1}/{num_steps}...", end=" ")
                
                # Calculate targets for this step
                targets = {}
                for motor_name in working_motors:
                    if motor_name in step_movements and current_positions[motor_name] is not None:
                        delta = step_movements[motor_name]
                        targets[motor_name] = current_positions[motor_name] + delta
                
                # Move all motors together
                for motor_name, target in targets.items():
                    try:
                        self.bus.write("Goal_Position", motor_name, target, normalize=True)
                    except Exception as e:
                        print(f"\n      ⚠️ {motor_name}: {e}")
                
                time.sleep(step_delay)
                
                # Read new positions
                for motor_name in working_motors:
                    try:
                        current_positions[motor_name] = self.bus.read("Present_Position", motor_name, normalize=True)
                    except:
                        pass
                
                print("✓")
            
            # Show movement summary
            print("\n   📊 Movement summary (Phase 1):")
            for motor_name in working_motors:
                start = start_positions[motor_name]
                current = current_positions[motor_name]
                if start is not None and current is not None:
                    delta = current - start
                    unit = "%" if motor_name == "gripper" else "°"
                    status = "✅" if abs(delta) > 0.5 else "⚠️"
                    print(f"      {status} {motor_name}: {start:.1f} → {current:.1f} (Δ={delta:+.1f}{unit})")
            
            time.sleep(0.5)
            
            # === PHASE 2: Return all motors to start position ===
            print(f"\n🔄 PHASE 2: Returning all motors to start positions...")
            self.status_bar.setText("🧪 Phase 2: Returning to start...")
            
            for step in range(num_steps):
                print(f"   Step {step + 1}/{num_steps}...", end=" ")
                
                # Calculate targets for this step (moving back toward start)
                targets = {}
                for motor_name in working_motors:
                    if motor_name in step_movements and current_positions[motor_name] is not None:
                        # Calculate step toward start position
                        start = start_positions[motor_name]
                        current = current_positions[motor_name]
                        remaining = start - current
                        steps_left = num_steps - step
                        if steps_left > 0:
                            step_size = remaining / steps_left
                            targets[motor_name] = current + step_size
                
                # Move all motors together
                for motor_name, target in targets.items():
                    try:
                        self.bus.write("Goal_Position", motor_name, target, normalize=True)
                    except Exception as e:
                        print(f"\n      ⚠️ {motor_name}: {e}")
                
                time.sleep(step_delay)
                
                # Read new positions
                for motor_name in working_motors:
                    try:
                        current_positions[motor_name] = self.bus.read("Present_Position", motor_name, normalize=True)
                    except:
                        pass
                
                print("✓")
            
            # Final position check
            print("\n   📊 Final positions:")
            all_returned = True
            for motor_name in working_motors:
                start = start_positions[motor_name]
                current = current_positions[motor_name]
                if start is not None and current is not None:
                    diff = abs(current - start)
                    unit = "%" if motor_name == "gripper" else "°"
                    status = "✅" if diff < 2.0 else "⚠️"
                    if diff >= 2.0:
                        all_returned = False
                    print(f"      {status} {motor_name}: {current:.1f}{unit} (diff from start: {diff:.1f}{unit})")
            
            if all_returned:
                print("\n✅ All motors returned to start positions")
            else:
                print("\n⚠️ Some motors didn't fully return - may need manual adjustment")
            
            print("\n✅ Coordinated motor testing complete")
            self.status_bar.setText("✅ Motor testing complete")
            
        except Exception as e:
            print(f"❌ Motor test failed: {e}")
            self.status_bar.setText(f"❌ Motor test failed: {e}")
        
        finally:
            # Resume monitoring
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                self.monitor_thread.paused = False
                print("▶️ Monitoring resumed")
    
    def open_manual_control(self):
        """Open the manual motor control dialog."""
        if not hasattr(self, 'bus') or not self.bus.is_connected:
            self.status_bar.setText("❌ Robot not connected")
            return
        
        # Pause monitoring during manual control
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            self.monitor_thread.paused = True
        
        try:
            dialog = ManualMotorControlDialog(self.bus, self.all_motors, self)
            dialog.exec()
        finally:
            # Resume monitoring after dialog closes
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                self.monitor_thread.paused = False
    
    def open_position_calibration(self):
        """Open the position calibration dialog for saving/recalling arm positions."""
        if not hasattr(self, 'bus') or not self.bus.is_connected:
            self.status_bar.setText("❌ Robot not connected")
            return
        
        # Pause monitoring during calibration
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            self.monitor_thread.paused = True
        
        try:
            dialog = PositionCalibrationDialog(self.bus, self.all_motors, self)
            dialog.exec()
        finally:
            # Resume monitoring after dialog closes
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                self.monitor_thread.paused = False
    
    def refresh_status(self):
        """Force refresh of all displays."""
        self.status_bar.setText("🔄 Refreshing all systems...")
        # The monitoring thread will update automatically
        QTimer.singleShot(1000, lambda: self.status_bar.setText("✅ Refresh complete"))
    
    def restart_app(self):
        """Restart the application."""
        import sys
        import os
        self.status_bar.setText("🔄 Restarting application...")
        
        # Stop monitoring thread
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.stop()
            self.monitor_thread.wait()
        
        # Close the window
        self.close()
        
        # Restart the application
        python = sys.executable
        os.execl(python, python, *sys.argv)
    
    def start_calibration(self):
        """Start physical chess board calibration by moving gripper to corners."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
        from PySide6.QtCore import QTimer
        
        class PhysicalCalibrationDialog(QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Physical Chess Board Calibration")
                self.setModal(True)
                self.corners = []
                self.corner_names = ["a1", "h1", "h8", "a8"]
                self.current_corner_idx = 0
                self.parent_ui = parent
                
                layout = QVBoxLayout(self)
                layout.setSpacing(20)
                layout.setContentsMargins(20, 20, 20, 20)
                
                # Instructions
                self.instruction_label = QLabel(
                    f"Step {self.current_corner_idx + 1}/4:\n\n"
                    f"Manually move the gripper to corner: {self.corner_names[0]}\n"
                    f"Then click RECORD POSITION"
                )
                self.instruction_label.setStyleSheet("""
                    font-size: 14pt;
                    font-weight: bold;
                    color: #58a6ff;
                    padding: 20px;
                    background: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 6px;
                """)
                self.instruction_label.setAlignment(Qt.AlignCenter)
                self.instruction_label.setWordWrap(True)
                layout.addWidget(self.instruction_label)
                
                # Current position display
                self.position_label = QLabel("Current motor positions:\nWaiting...")
                self.position_label.setStyleSheet("""
                    font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
                    font-size: 11pt;
                    color: #8b949e;
                    padding: 15px;
                    background: #0d1117;
                    border: 1px solid #30363d;
                    border-radius: 4px;
                """)
                layout.addWidget(self.position_label)
                
                # Update timer
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.update_current_position)
                self.timer.start(100)  # Update 10 times per second
                
                # Button row
                button_row = QHBoxLayout()
                
                self.record_btn = QPushButton("RECORD POSITION")
                self.record_btn.clicked.connect(self.record_corner)
                self.record_btn.setStyleSheet("""
                    QPushButton {
                        background: #238636;
                        color: white;
                        font-weight: bold;
                        font-size: 12pt;
                        padding: 12px 24px;
                        border-radius: 6px;
                    }
                    QPushButton:hover { background: #2ea043; }
                """)
                button_row.addWidget(self.record_btn)
                
                undo_btn = QPushButton("Undo Last")
                undo_btn.clicked.connect(self.undo_last)
                button_row.addWidget(undo_btn)
                
                cancel_btn = QPushButton("Cancel")
                cancel_btn.clicked.connect(self.reject)
                button_row.addWidget(cancel_btn)
                
                layout.addLayout(button_row)
                
                self.resize(600, 400)
            
            def update_current_position(self):
                """Update display with current motor positions."""
                try:
                    positions = {}
                    for motor_name in self.parent_ui.all_motors.keys():
                        try:
                            pos = self.parent_ui.bus.read("Present_Position", motor_name, normalize=True)
                            positions[motor_name] = f"{pos:.1f}°"
                        except:
                            positions[motor_name] = "N/A"
                    
                    pos_text = "Current motor positions:\n" + "\n".join([f"{k}: {v}" for k, v in positions.items()])
                    self.position_label.setText(pos_text)
                except:
                    pass
            
            def record_corner(self):
                """Record current motor positions for this corner."""
                try:
                    # Read all motor positions
                    positions = {}
                    for motor_name in self.parent_ui.all_motors.keys():
                        pos = self.parent_ui.bus.read("Present_Position", motor_name, normalize=True)
                        positions[motor_name] = pos
                    
                    # Calculate XYZ using forward kinematics
                    if self.parent_ui.kinematics:
                        joint_angles = [positions[name] for name in self.parent_ui.all_motors.keys()]
                        xyz = self.parent_ui.kinematics.forward(np.deg2rad(joint_angles))[:3]
                    else:
                        # Fallback: use simple approximation
                        xyz = np.array([0.0, 0.0, 0.0])
                    
                    self.corners.append({
                        'name': self.corner_names[self.current_corner_idx],
                        'motors': positions,
                        'xyz': xyz
                    })
                    
                    self.current_corner_idx += 1
                    
                    if self.current_corner_idx < 4:
                        self.instruction_label.setText(
                            f"Step {self.current_corner_idx + 1}/4:\n\n"
                            f"Manually move the gripper to corner: {self.corner_names[self.current_corner_idx]}\n"
                            f"Then click RECORD POSITION"
                        )
                    else:
                        self.timer.stop()
                        self.instruction_label.setText(
                            "✅ All 4 corners recorded!\n\n"
                            "Click SAVE to complete calibration"
                        )
                        self.record_btn.setText("SAVE CALIBRATION")
                        self.record_btn.clicked.disconnect()
                        self.record_btn.clicked.connect(self.accept)
                except Exception as e:
                    self.instruction_label.setText(f"❌ Error recording position: {e}")
            
            def undo_last(self):
                """Undo the last recorded corner."""
                if self.corners:
                    self.corners.pop()
                    self.current_corner_idx -= 1
                    self.instruction_label.setText(
                        f"Step {self.current_corner_idx + 1}/4:\n\n"
                        f"Manually move the gripper to corner: {self.corner_names[self.current_corner_idx]}\n"
                        f"Then click RECORD POSITION"
                    )
                    if self.record_btn.text() == "SAVE CALIBRATION":
                        self.record_btn.setText("RECORD POSITION")
                        self.record_btn.clicked.disconnect()
                        self.record_btn.clicked.connect(self.record_corner)
                        self.timer.start(100)
        
        # Get current state
        if not hasattr(self, 'bus') or not self.bus:
            self.status_bar.setText("❌ Robot not connected")
            return
        
        dialog = PhysicalCalibrationDialog(self)
        if dialog.exec() == QDialog.Accepted:
            # Save calibration
            self.save_board_calibration_physical(dialog.corners)
    
    def save_board_calibration_physical(self, corners_data):
        """Save chess board calibration from physical corner measurements."""
        try:
            from lerobot.perception.chess.board_model import BoardModel
            from lerobot.configs.chessboard import ChessBoardParams
            from lerobot.utils.geometry import SE3
            
            # Extract XYZ coordinates from corners
            corners_xyz = np.array([c['xyz'] for c in corners_data])
            
            # Define board frame: a1 at origin, h1 along +X, a8 along +Y
            # 8 squares × 50mm = 400mm = 0.4m
            board_corners_ideal = np.array([
                [0.0, 0.0, 0.0],      # a1
                [0.4, 0.0, 0.0],      # h1
                [0.4, 0.4, 0.0],      # h8
                [0.0, 0.4, 0.0],      # a8
            ])
            
            # Calculate rigid transform from board frame to robot base frame
            # Using Procrustes / least-squares rigid alignment
            centroid_board = board_corners_ideal.mean(axis=0)
            centroid_robot = corners_xyz.mean(axis=0)
            
            # Center the points
            board_centered = board_corners_ideal - centroid_board
            robot_centered = corners_xyz - centroid_robot
            
            # Compute rotation using SVD
            H = board_centered.T @ robot_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = centroid_robot - R @ centroid_board
            
            # Create SE3 transform
            T_base_board = SE3.from_rt(R, t)
            
            # Create board model
            board_model = BoardModel(
                params=ChessBoardParams(square_size_mm=50.0),
                T_base_board=T_base_board
            )
            
            # Save to calibration directory
            calib_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
            calib_dir.mkdir(parents=True, exist_ok=True)
            out_path = calib_dir / "chess_board_model.json"
            board_model.save(out_path)
            
            # Also save motor positions for reference
            motor_calib_path = calib_dir / "chess_corner_motors.json"
            with open(motor_calib_path, 'w') as f:
                json.dump(corners_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            
            self.status_bar.setText(f"✅ Physical calibration saved! LLM can now control chess positions.")
            print(f"✅ Chess board physical calibration completed")
            print(f"   Board model: {out_path}")
            print(f"   Motor positions: {motor_calib_path}")
        except Exception as e:
            self.status_bar.setText(f"❌ Calibration save failed: {e}")
            print(f"❌ Calibration error: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_file_watcher(self):
        """Setup file watcher for hot reload in development mode."""
        try:
            import sys
            import os
            
            self.file_watcher = QFileSystemWatcher()
            # Watch the current Python file
            current_file = os.path.abspath(sys.argv[0])
            if os.path.exists(current_file):
                self.file_watcher.addPath(current_file)
                self.file_watcher.fileChanged.connect(self.on_file_changed)
                print(f"👀 Watching {current_file} for changes (dev mode)")
        except Exception as e:
            print(f"⚠️ Could not setup file watcher: {e}")
    
    def on_file_changed(self, path):
        """Called when the watched file changes."""
        if self.dev_mode:
            # Update status bar with restart notification - terminal style
            self.status_bar.setText("> Code changed! Restart app to see changes.")
            self.status_bar.setStyleSheet("color: #f0883e;")
            print(f"> File changed: {path}")
            print("  Please restart the app manually to see changes.")


class ManualMotorControlDialog(QDialog):
    """Dialog for manual motor control with sliders and direct position input."""
    
    def __init__(self, bus, motors, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.motors = motors
        self.parent_window = parent
        self.setWindowTitle("Manual Motor Control")
        self.setMinimumSize(800, 700)
        self.setStyleSheet("""
            QDialog {
                background: #0d1117;
                color: #c9d1d9;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                background: #161b22;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #c9d1d9;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #21262d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #58a6ff;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #79c0ff;
            }
            QPushButton {
                background: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #30363d;
                border-color: #8b949e;
            }
            QPushButton:pressed {
                background: #0d1117;
            }
            QSpinBox, QDoubleSpinBox {
                background: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """)
        
        self.setup_ui()
        self.start_position_updates()
    
    def setup_ui(self):
        """Create the manual control UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("🎮 Manual Motor Control")
        title.setStyleSheet("font-size: 18pt; font-weight: bold; color: #58a6ff;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel("Move sliders or enter values to control each motor. Use RAW mode for direct position control.")
        instructions.setStyleSheet("color: #8b949e; font-size: 10pt;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Scroll area for motor controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)
        
        # Motor controls storage
        self.motor_controls = {}
        
        # Create controls for each motor
        motor_configs = {
            "shoulder_pan": {"min": -180, "max": 180, "unit": "°", "color": "#e74c3c"},
            "shoulder_lift": {"min": -180, "max": 180, "unit": "°", "color": "#e67e22"},
            "elbow_flex": {"min": -180, "max": 180, "unit": "°", "color": "#f1c40f"},
            "wrist_flex": {"min": -180, "max": 180, "unit": "°", "color": "#2ecc71"},
            "wrist_roll": {"min": -180, "max": 180, "unit": "°", "color": "#3498db"},
            "gripper": {"min": 0, "max": 100, "unit": "%", "color": "#9b59b6"},
        }
        
        for motor_name in self.motors.keys():
            config = motor_configs.get(motor_name, {"min": -180, "max": 180, "unit": "°", "color": "#58a6ff"})
            motor_group = self.create_motor_control(motor_name, config)
            scroll_layout.addWidget(motor_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        # All torque toggle
        self.torque_all_btn = QPushButton("🔌 DISABLE ALL TORQUE")
        self.torque_all_btn.clicked.connect(self.toggle_all_torque)
        self.torque_all_btn.setStyleSheet("""
            QPushButton {
                background: #da3633;
                color: white;
            }
            QPushButton:hover {
                background: #f85149;
            }
        """)
        button_layout.addWidget(self.torque_all_btn)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 REFRESH POSITIONS")
        refresh_btn.clicked.connect(self.refresh_all_positions)
        button_layout.addWidget(refresh_btn)
        
        # Center all button
        center_btn = QPushButton("🎯 CENTER ALL")
        center_btn.clicked.connect(self.center_all_motors)
        center_btn.setStyleSheet("""
            QPushButton {
                background: #1f6feb;
                color: white;
            }
            QPushButton:hover {
                background: #388bfd;
            }
        """)
        button_layout.addWidget(center_btn)
        
        button_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("✖ CLOSE")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_motor_control(self, motor_name: str, config: dict) -> QGroupBox:
        """Create control group for a single motor."""
        group = QGroupBox(f"Motor: {motor_name}")
        group.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {config['color']};
                border-radius: 8px;
            }}
            QGroupBox::title {{
                color: {config['color']};
            }}
        """)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Row 1: Current position display
        pos_row = QHBoxLayout()
        
        pos_label = QLabel("Position:")
        pos_label.setStyleSheet("font-weight: bold;")
        pos_row.addWidget(pos_label)
        
        norm_pos_label = QLabel("--")
        norm_pos_label.setStyleSheet(f"color: {config['color']}; font-size: 14pt; font-weight: bold; min-width: 80px;")
        pos_row.addWidget(norm_pos_label)
        
        pos_row.addWidget(QLabel(config['unit']))
        
        pos_row.addSpacing(20)
        
        pos_row.addWidget(QLabel("Raw:"))
        raw_pos_label = QLabel("--")
        raw_pos_label.setStyleSheet("color: #8b949e; min-width: 60px;")
        pos_row.addWidget(raw_pos_label)
        
        pos_row.addStretch()
        
        # Torque indicator
        torque_label = QLabel("⚡ ON")
        torque_label.setStyleSheet("color: #3fb950; font-weight: bold;")
        pos_row.addWidget(torque_label)
        
        layout.addLayout(pos_row)
        
        # Row 2: Slider control
        slider_row = QHBoxLayout()
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(config['min'] * 10))  # Use 0.1 degree precision
        slider.setMaximum(int(config['max'] * 10))
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(int((config['max'] - config['min']) * 10 / 10))
        slider_row.addWidget(slider, 1)
        
        layout.addLayout(slider_row)
        
        # Row 3: Direct input controls
        input_row = QHBoxLayout()
        
        input_row.addWidget(QLabel("Target:"))
        
        # Normalized input
        norm_spin = QDoubleSpinBox()
        norm_spin.setRange(config['min'], config['max'])
        norm_spin.setDecimals(1)
        norm_spin.setSuffix(f" {config['unit']}")
        norm_spin.setMinimumWidth(100)
        input_row.addWidget(norm_spin)
        
        # Go button for normalized
        go_norm_btn = QPushButton("GO")
        go_norm_btn.setStyleSheet(f"""
            QPushButton {{
                background: {config['color']};
                color: white;
                padding: 4px 12px;
            }}
        """)
        go_norm_btn.clicked.connect(lambda: self.move_motor_normalized(motor_name, norm_spin.value()))
        input_row.addWidget(go_norm_btn)
        
        input_row.addSpacing(20)
        
        # Raw input
        input_row.addWidget(QLabel("Raw:"))
        raw_spin = QSpinBox()
        raw_spin.setRange(0, 4095)
        raw_spin.setMinimumWidth(80)
        input_row.addWidget(raw_spin)
        
        # Go button for raw
        go_raw_btn = QPushButton("GO RAW")
        go_raw_btn.setStyleSheet("""
            QPushButton {
                background: #f0883e;
                color: white;
                padding: 4px 12px;
            }
        """)
        go_raw_btn.clicked.connect(lambda: self.move_motor_raw(motor_name, raw_spin.value()))
        input_row.addWidget(go_raw_btn)
        
        input_row.addStretch()
        
        # Toggle torque button
        torque_btn = QPushButton("Toggle Torque")
        torque_btn.clicked.connect(lambda: self.toggle_motor_torque(motor_name))
        input_row.addWidget(torque_btn)
        
        layout.addLayout(input_row)
        
        # Row 4: Quick actions
        quick_row = QHBoxLayout()
        
        # Nudge buttons
        nudge_minus_10 = QPushButton("-10")
        nudge_minus_10.clicked.connect(lambda: self.nudge_motor(motor_name, -10))
        quick_row.addWidget(nudge_minus_10)
        
        nudge_minus_5 = QPushButton("-5")
        nudge_minus_5.clicked.connect(lambda: self.nudge_motor(motor_name, -5))
        quick_row.addWidget(nudge_minus_5)
        
        nudge_minus_1 = QPushButton("-1")
        nudge_minus_1.clicked.connect(lambda: self.nudge_motor(motor_name, -1))
        quick_row.addWidget(nudge_minus_1)
        
        nudge_plus_1 = QPushButton("+1")
        nudge_plus_1.clicked.connect(lambda: self.nudge_motor(motor_name, 1))
        quick_row.addWidget(nudge_plus_1)
        
        nudge_plus_5 = QPushButton("+5")
        nudge_plus_5.clicked.connect(lambda: self.nudge_motor(motor_name, 5))
        quick_row.addWidget(nudge_plus_5)
        
        nudge_plus_10 = QPushButton("+10")
        nudge_plus_10.clicked.connect(lambda: self.nudge_motor(motor_name, 10))
        quick_row.addWidget(nudge_plus_10)
        
        quick_row.addStretch()
        
        # Raw nudge buttons (for debugging motor 3)
        quick_row.addWidget(QLabel("Raw:"))
        
        raw_minus_100 = QPushButton("-100")
        raw_minus_100.setStyleSheet("background: #21262d;")
        raw_minus_100.clicked.connect(lambda: self.nudge_motor_raw(motor_name, -100))
        quick_row.addWidget(raw_minus_100)
        
        raw_plus_100 = QPushButton("+100")
        raw_plus_100.setStyleSheet("background: #21262d;")
        raw_plus_100.clicked.connect(lambda: self.nudge_motor_raw(motor_name, 100))
        quick_row.addWidget(raw_plus_100)
        
        layout.addLayout(quick_row)
        
        # Store controls for this motor
        self.motor_controls[motor_name] = {
            "norm_pos_label": norm_pos_label,
            "raw_pos_label": raw_pos_label,
            "torque_label": torque_label,
            "slider": slider,
            "norm_spin": norm_spin,
            "raw_spin": raw_spin,
            "config": config,
        }
        
        # Connect slider to movement
        slider.sliderReleased.connect(lambda m=motor_name: self.slider_released(m))
        
        return group
    
    def start_position_updates(self):
        """Start timer to update position displays."""
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_positions)
        self.update_timer.start(200)  # Update 5 times per second
        self.update_positions()  # Initial update
    
    def update_positions(self):
        """Update all motor position displays."""
        for motor_name, controls in self.motor_controls.items():
            try:
                # Read normalized position
                norm_pos = self.bus.read("Present_Position", motor_name, normalize=True)
                controls["norm_pos_label"].setText(f"{norm_pos:.1f}")
                
                # Update slider (without triggering signal)
                controls["slider"].blockSignals(True)
                controls["slider"].setValue(int(norm_pos * 10))
                controls["slider"].blockSignals(False)
                
                # Read raw position
                raw_pos = self.bus.read("Present_Position", motor_name, normalize=False)
                controls["raw_pos_label"].setText(str(raw_pos))
                controls["raw_spin"].setValue(raw_pos)
                
                # Update torque status
                try:
                    torque = self.bus.read("Torque_Enable", motor_name)
                    if torque == 1:
                        controls["torque_label"].setText("⚡ ON")
                        controls["torque_label"].setStyleSheet("color: #3fb950; font-weight: bold;")
                    else:
                        controls["torque_label"].setText("⚡ OFF")
                        controls["torque_label"].setStyleSheet("color: #f85149; font-weight: bold;")
                except:
                    pass
                
            except Exception as e:
                controls["norm_pos_label"].setText("ERR")
                controls["raw_pos_label"].setText("ERR")
    
    def slider_released(self, motor_name: str):
        """Handle slider release - move motor to slider position."""
        controls = self.motor_controls[motor_name]
        target = controls["slider"].value() / 10.0
        self.move_motor_normalized(motor_name, target)
    
    def move_motor_normalized(self, motor_name: str, target: float):
        """Move motor to normalized position."""
        try:
            print(f"🎯 Moving {motor_name} to {target:.1f}° (normalized)")
            self.bus.write("Goal_Position", motor_name, target, normalize=True)
        except Exception as e:
            print(f"❌ Error moving {motor_name}: {e}")
    
    def move_motor_raw(self, motor_name: str, raw_target: int):
        """Move motor to raw position."""
        try:
            print(f"🎯 Moving {motor_name} to {raw_target} (RAW)")
            self.bus.write("Goal_Position", motor_name, raw_target, normalize=False)
        except Exception as e:
            print(f"❌ Error moving {motor_name}: {e}")
    
    def nudge_motor(self, motor_name: str, delta: float):
        """Nudge motor by delta degrees."""
        try:
            current = self.bus.read("Present_Position", motor_name, normalize=True)
            target = current + delta
            print(f"🔧 Nudging {motor_name}: {current:.1f}° + {delta}° = {target:.1f}°")
            self.bus.write("Goal_Position", motor_name, target, normalize=True)
        except Exception as e:
            print(f"❌ Error nudging {motor_name}: {e}")
    
    def nudge_motor_raw(self, motor_name: str, delta: int):
        """Nudge motor by delta raw units."""
        try:
            current = self.bus.read("Present_Position", motor_name, normalize=False)
            target = current + delta
            print(f"🔧 Nudging {motor_name} RAW: {current} + {delta} = {target}")
            self.bus.write("Goal_Position", motor_name, target, normalize=False)
        except Exception as e:
            print(f"❌ Error nudging {motor_name} (raw): {e}")
    
    def toggle_motor_torque(self, motor_name: str):
        """Toggle torque for a single motor."""
        try:
            current = self.bus.read("Torque_Enable", motor_name)
            new_state = 0 if current == 1 else 1
            self.bus.write("Torque_Enable", motor_name, new_state)
            print(f"⚡ {motor_name} torque: {'ON' if new_state else 'OFF'}")
        except Exception as e:
            print(f"❌ Error toggling torque for {motor_name}: {e}")
    
    def toggle_all_torque(self):
        """Toggle torque for all motors."""
        try:
            # Check if any motor has torque enabled
            any_enabled = False
            for motor_name in self.motors.keys():
                try:
                    if self.bus.read("Torque_Enable", motor_name) == 1:
                        any_enabled = True
                        break
                except:
                    pass
            
            # Toggle all motors
            new_state = 0 if any_enabled else 1
            for motor_name in self.motors.keys():
                try:
                    self.bus.write("Torque_Enable", motor_name, new_state)
                except:
                    pass
            
            # Update button text
            if new_state == 0:
                self.torque_all_btn.setText("🔌 ENABLE ALL TORQUE")
                self.torque_all_btn.setStyleSheet("""
                    QPushButton {
                        background: #238636;
                        color: white;
                    }
                    QPushButton:hover {
                        background: #2ea043;
                    }
                """)
            else:
                self.torque_all_btn.setText("🔌 DISABLE ALL TORQUE")
                self.torque_all_btn.setStyleSheet("""
                    QPushButton {
                        background: #da3633;
                        color: white;
                    }
                    QPushButton:hover {
                        background: #f85149;
                    }
                """)
            
            print(f"⚡ All motors torque: {'ON' if new_state else 'OFF'}")
        except Exception as e:
            print(f"❌ Error toggling all torque: {e}")
    
    def refresh_all_positions(self):
        """Force refresh all position displays."""
        self.update_positions()
        print("🔄 Positions refreshed")
    
    def center_all_motors(self):
        """Move all motors to center position (0 degrees or 50%)."""
        for motor_name in self.motors.keys():
            try:
                if motor_name == "gripper":
                    target = 50  # 50% for gripper
                else:
                    target = 0  # 0 degrees for joints
                self.bus.write("Goal_Position", motor_name, target, normalize=True)
                time.sleep(0.3)
            except Exception as e:
                print(f"❌ Error centering {motor_name}: {e}")
        print("🎯 All motors centered")
    
    def closeEvent(self, event):
        """Clean up when dialog closes."""
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        super().closeEvent(event)


class PositionCalibrationDialog(QDialog):
    """Dialog for saving and recalling arm positions."""
    
    POSITIONS_FILE = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/saved_positions.json"
    
    # Default preset positions for chess robot
    DEFAULT_POSITIONS = {
        "home": {
            "description": "Safe home position",
            "positions": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 50.0
            }
        },
        "overview": {
            "description": "Overview position to see chess board",
            "positions": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 40.0,
                "elbow_flex": 40.0,
                "wrist_flex": -30.0,
                "wrist_roll": 0.0,
                "gripper": 50.0
            }
        },
        "ready": {
            "description": "Ready position above board center",
            "positions": {
                "shoulder_pan": 0.0,
                "shoulder_lift": -20.0,
                "elbow_flex": -20.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 30.0
            }
        }
    }
    
    def __init__(self, bus, motors, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.motors = motors
        self.parent_window = parent
        self.setWindowTitle("Position Calibration")
        self.setMinimumSize(900, 700)
        self.setStyleSheet("""
            QDialog {
                background: #0d1117;
                color: #c9d1d9;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                background: #161b22;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #c9d1d9;
            }
            QPushButton {
                background: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #30363d;
                border-color: #8b949e;
            }
            QPushButton:pressed {
                background: #0d1117;
            }
            QLineEdit {
                background: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 8px;
            }
            QListWidget {
                background: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #21262d;
            }
            QListWidget::item:selected {
                background: #238636;
            }
            QListWidget::item:hover {
                background: #30363d;
            }
        """)
        
        self.saved_positions = self.load_positions()
        self.setup_ui()
        self.start_position_updates()
    
    def load_positions(self) -> dict:
        """Load saved positions from file."""
        if self.POSITIONS_FILE.exists():
            try:
                with open(self.POSITIONS_FILE) as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading positions: {e}")
        
        # Return default positions if file doesn't exist
        return self.DEFAULT_POSITIONS.copy()
    
    def save_positions_to_file(self):
        """Save positions to file."""
        try:
            self.POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.POSITIONS_FILE, 'w') as f:
                json.dump(self.saved_positions, f, indent=2)
            print(f"✅ Positions saved to {self.POSITIONS_FILE}")
        except Exception as e:
            print(f"❌ Error saving positions: {e}")
    
    def setup_ui(self):
        """Create the position calibration UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("📍 Position Calibration")
        title.setStyleSheet("font-size: 18pt; font-weight: bold; color: #238636;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Save arm positions for quick recall. Disable torque to manually position the arm, "
            "then save. Use 'GO TO' to move to a saved position."
        )
        instructions.setStyleSheet("color: #8b949e; font-size: 10pt;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Main content - two columns
        content_layout = QHBoxLayout()
        
        # LEFT: Current position and save controls
        left_group = QGroupBox("Current Position")
        left_layout = QVBoxLayout(left_group)
        
        # Current position display
        self.position_labels = {}
        pos_grid = QGridLayout()
        pos_grid.setSpacing(8)
        
        row = 0
        for motor_name in self.motors.keys():
            name_label = QLabel(f"{motor_name}:")
            name_label.setStyleSheet("font-weight: bold;")
            pos_grid.addWidget(name_label, row, 0)
            
            value_label = QLabel("--")
            value_label.setStyleSheet("color: #58a6ff; font-size: 12pt;")
            pos_grid.addWidget(value_label, row, 1)
            
            unit_label = QLabel("%" if motor_name == "gripper" else "°")
            pos_grid.addWidget(unit_label, row, 2)
            
            self.position_labels[motor_name] = value_label
            row += 1
        
        left_layout.addLayout(pos_grid)
        
        # Torque control
        left_layout.addSpacing(10)
        self.torque_btn = QPushButton("🔌 DISABLE TORQUE (Manual Position)")
        self.torque_btn.clicked.connect(self.toggle_torque)
        self.torque_btn.setStyleSheet("""
            QPushButton {
                background: #da3633;
                color: white;
            }
            QPushButton:hover {
                background: #f85149;
            }
        """)
        left_layout.addWidget(self.torque_btn)
        
        # Save section
        left_layout.addSpacing(20)
        save_label = QLabel("Save Current Position:")
        save_label.setStyleSheet("font-weight: bold; color: #238636;")
        left_layout.addWidget(save_label)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., board_center, pick_e4")
        name_layout.addWidget(self.name_input)
        left_layout.addLayout(name_layout)
        
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.desc_input = QLineEdit()
        self.desc_input.setPlaceholderText("Optional description")
        desc_layout.addWidget(self.desc_input)
        left_layout.addLayout(desc_layout)
        
        self.save_btn = QPushButton("💾 SAVE POSITION")
        self.save_btn.clicked.connect(self.save_current_position)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #238636;
                color: white;
                font-size: 12pt;
                padding: 12px;
            }
            QPushButton:hover {
                background: #2ea043;
            }
        """)
        left_layout.addWidget(self.save_btn)
        
        left_layout.addStretch()
        content_layout.addWidget(left_group, 1)
        
        # RIGHT: Saved positions list
        right_group = QGroupBox("Saved Positions")
        right_layout = QVBoxLayout(right_group)
        
        # Position list
        self.position_list = QListWidget()
        self.position_list.itemClicked.connect(self.on_position_selected)
        self.position_list.itemDoubleClicked.connect(self.go_to_selected)
        self.refresh_position_list()
        right_layout.addWidget(self.position_list, 1)
        
        # Selected position info
        self.selected_info = QLabel("Select a position to see details")
        self.selected_info.setStyleSheet("color: #8b949e; font-size: 9pt;")
        self.selected_info.setWordWrap(True)
        right_layout.addWidget(self.selected_info)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.goto_btn = QPushButton("🎯 GO TO")
        self.goto_btn.clicked.connect(self.go_to_selected)
        self.goto_btn.setStyleSheet("""
            QPushButton {
                background: #1f6feb;
                color: white;
            }
            QPushButton:hover {
                background: #388bfd;
            }
        """)
        btn_layout.addWidget(self.goto_btn)
        
        self.update_btn = QPushButton("🔄 UPDATE")
        self.update_btn.setToolTip("Update selected position with current arm position")
        self.update_btn.clicked.connect(self.update_selected_position)
        self.update_btn.setStyleSheet("""
            QPushButton {
                background: #f0883e;
                color: white;
            }
            QPushButton:hover {
                background: #ff9a4d;
            }
        """)
        btn_layout.addWidget(self.update_btn)
        
        self.delete_btn = QPushButton("🗑️ DELETE")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background: #da3633;
                color: white;
            }
            QPushButton:hover {
                background: #f85149;
            }
        """)
        btn_layout.addWidget(self.delete_btn)
        
        right_layout.addLayout(btn_layout)
        
        # Quick preset buttons
        right_layout.addSpacing(10)
        preset_label = QLabel("Quick Presets:")
        preset_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(preset_label)
        
        preset_layout = QHBoxLayout()
        for preset_name in ["home", "overview", "ready"]:
            if preset_name in self.saved_positions:
                btn = QPushButton(preset_name.title())
                btn.clicked.connect(lambda checked, n=preset_name: self.go_to_position(n))
                preset_layout.addWidget(btn)
        preset_layout.addStretch()
        right_layout.addLayout(preset_layout)
        
        content_layout.addWidget(right_group, 1)
        layout.addLayout(content_layout, 1)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 REFRESH")
        refresh_btn.clicked.connect(self.refresh_position_list)
        bottom_layout.addWidget(refresh_btn)
        
        bottom_layout.addStretch()
        
        close_btn = QPushButton("✖ CLOSE")
        close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(close_btn)
        
        layout.addLayout(bottom_layout)
    
    def start_position_updates(self):
        """Start timer to update position displays."""
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_positions)
        self.update_timer.start(200)
        self.update_positions()
    
    def update_positions(self):
        """Update current position display."""
        for motor_name, label in self.position_labels.items():
            try:
                pos = self.bus.read("Present_Position", motor_name, normalize=True)
                label.setText(f"{pos:.1f}")
            except:
                label.setText("ERR")
    
    def toggle_torque(self):
        """Toggle torque on/off for manual positioning."""
        try:
            # Check current state
            any_enabled = False
            for motor_name in self.motors.keys():
                try:
                    if self.bus.read("Torque_Enable", motor_name) == 1:
                        any_enabled = True
                        break
                except:
                    pass
            
            # Toggle
            new_state = 0 if any_enabled else 1
            for motor_name in self.motors.keys():
                try:
                    self.bus.write("Torque_Enable", motor_name, new_state)
                except:
                    pass
            
            # Update button
            if new_state == 0:
                self.torque_btn.setText("🔌 ENABLE TORQUE")
                self.torque_btn.setStyleSheet("""
                    QPushButton {
                        background: #238636;
                        color: white;
                    }
                    QPushButton:hover {
                        background: #2ea043;
                    }
                """)
                print("⚡ Torque DISABLED - you can manually position the arm")
            else:
                self.torque_btn.setText("🔌 DISABLE TORQUE (Manual Position)")
                self.torque_btn.setStyleSheet("""
                    QPushButton {
                        background: #da3633;
                        color: white;
                    }
                    QPushButton:hover {
                        background: #f85149;
                    }
                """)
                print("⚡ Torque ENABLED - motors active")
        except Exception as e:
            print(f"❌ Error toggling torque: {e}")
    
    def save_current_position(self):
        """Save the current arm position."""
        name = self.name_input.text().strip()
        if not name:
            print("⚠️ Please enter a position name")
            return
        
        # Sanitize name
        name = name.lower().replace(" ", "_")
        
        # Check if position already exists and ask for confirmation
        if name in self.saved_positions:
            reply = QMessageBox.question(
                self,
                "Overwrite Position?",
                f"Position '{name}' already exists.\n\nDo you want to overwrite it with the current arm position?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                print(f"⚠️ Overwrite cancelled for '{name}'")
                return
        
        # Read current positions
        positions = {}
        for motor_name in self.motors.keys():
            try:
                pos = self.bus.read("Present_Position", motor_name, normalize=True)
                positions[motor_name] = round(pos, 1)
            except Exception as e:
                print(f"⚠️ Could not read {motor_name}: {e}")
                return
        
        # Save (will overwrite if exists)
        is_update = name in self.saved_positions
        self.saved_positions[name] = {
            "description": self.desc_input.text().strip() or f"Saved position: {name}",
            "positions": positions
        }
        
        self.save_positions_to_file()
        self.refresh_position_list()
        
        # Clear inputs
        self.name_input.clear()
        self.desc_input.clear()
        
        if is_update:
            print(f"✅ Position '{name}' updated")
        else:
            print(f"✅ Position '{name}' saved")
    
    def refresh_position_list(self):
        """Refresh the list of saved positions."""
        self.position_list.clear()
        for name, data in self.saved_positions.items():
            desc = data.get("description", "")
            self.position_list.addItem(f"📍 {name}: {desc[:40]}...")
    
    def on_position_selected(self, item):
        """Show details of selected position."""
        text = item.text()
        name = text.split(":")[0].replace("📍 ", "").strip()
        
        if name in self.saved_positions:
            data = self.saved_positions[name]
            info = f"Position: {name}\n"
            info += f"Description: {data.get('description', 'N/A')}\n\n"
            info += "Joint positions:\n"
            for motor, pos in data.get("positions", {}).items():
                unit = "%" if motor == "gripper" else "°"
                info += f"  {motor}: {pos}{unit}\n"
            self.selected_info.setText(info)
    
    def go_to_selected(self):
        """Go to the selected position."""
        item = self.position_list.currentItem()
        if not item:
            return
        
        text = item.text()
        name = text.split(":")[0].replace("📍 ", "").strip()
        self.go_to_position(name)
    
    def go_to_position(self, name: str):
        """Move arm to a saved position using coordinated movement."""
        if name not in self.saved_positions:
            print(f"⚠️ Position '{name}' not found")
            return
        
        data = self.saved_positions[name]
        positions = data.get("positions", {})
        
        print(f"🎯 Moving to position '{name}'...")
        
        try:
            # Use sync_write for coordinated movement
            self.bus.sync_write("Goal_Position", positions, normalize=True)
            print(f"✅ Moved to '{name}'")
        except Exception as e:
            print(f"⚠️ sync_write failed: {e}, trying sequential...")
            # Fall back to sequential
            for motor_name, target in positions.items():
                try:
                    self.bus.write("Goal_Position", motor_name, target, normalize=True)
                    time.sleep(0.1)
                except Exception as e2:
                    print(f"⚠️ {motor_name}: {e2}")
    
    def update_selected_position(self):
        """Update the selected position with current arm position."""
        item = self.position_list.currentItem()
        if not item:
            print("⚠️ Please select a position to update")
            return
        
        text = item.text()
        name = text.split(":")[0].replace("📍 ", "").strip()
        
        if name not in self.saved_positions:
            print(f"⚠️ Position '{name}' not found")
            return
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Update Position?",
            f"Update position '{name}' with the current arm position?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return
        
        # Read current positions
        positions = {}
        for motor_name in self.motors.keys():
            try:
                pos = self.bus.read("Present_Position", motor_name, normalize=True)
                positions[motor_name] = round(pos, 1)
            except Exception as e:
                print(f"⚠️ Could not read {motor_name}: {e}")
                return
        
        # Update the position (keep the existing description if not changed)
        old_desc = self.saved_positions[name].get("description", "")
        self.saved_positions[name] = {
            "description": old_desc,
            "positions": positions
        }
        
        self.save_positions_to_file()
        self.refresh_position_list()
        self.on_position_selected(item)  # Update the info display
        
        print(f"✅ Position '{name}' updated with current arm position")
    
    def delete_selected(self):
        """Delete the selected position."""
        item = self.position_list.currentItem()
        if not item:
            return
        
        text = item.text()
        name = text.split(":")[0].replace("📍 ", "").strip()
        
        if name in self.saved_positions:
            # Ask for confirmation
            reply = QMessageBox.question(
                self,
                "Delete Position?",
                f"Are you sure you want to delete position '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            
            del self.saved_positions[name]
            self.save_positions_to_file()
            self.refresh_position_list()
            self.selected_info.setText("Position deleted")
            print(f"🗑️ Position '{name}' deleted")
    
    def closeEvent(self, event):
        """Clean up when dialog closes."""
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        # Re-enable torque before closing
        for motor_name in self.motors.keys():
            try:
                self.bus.write("Torque_Enable", motor_name, 1)
            except:
                pass
        super().closeEvent(event)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Chess Robot Monitoring UI with ChatKit LLM Control")
    p.add_argument("--port", required=False, help="Robot serial port (optional - can be selected in UI)")
    p.add_argument("--dev", action="store_true", help="Enable development mode with file watching")
    p.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--workflow-id", type=str, help="ChatKit workflow ID from Agent Builder (or set OPENAI_CHATKIT_WORKFLOW_ID env var)")
    args = p.parse_args()
    
    app = QApplication([])
    
    try:
        ui = ChessRobotUILLM(args.port, dev_mode=args.dev, api_key=args.api_key, workflow_id=args.workflow_id)
        ui.show()
        print("🚀 Starting Chess Robot UI with ChatKit LLM Control...")
        if args.dev:
            print("🔧 Development mode: File watching enabled")
        if ui.llm_enabled:
            if ui.chatkit_session:
                print(f"✅ ChatKit enabled with workflow: {ui.workflow_id}")
            else:
                print("✅ LLM control enabled (using direct chat.completions)")
        else:
            print("⚠️ LLM control disabled - set OPENAI_API_KEY or use --api-key")
        print("Close the window or click 'Stop' to exit")
        app.exec()
    except Exception as e:
        print(f"❌ UI failed to start: {e}")
    finally:
        print("👋 Chess Robot UI closed")


if __name__ == "__main__":
    main()
