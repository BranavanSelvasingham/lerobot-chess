#!/usr/bin/env python

"""
Chess Robot Monitoring UI with LLM Control
Shows live camera feed, motor status, chess board diagram, and LLM-based natural language control.
"""

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QGroupBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QFrame, QToolTip, QScrollArea,
    QTextEdit, QLineEdit, QPlainTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize, QFileSystemWatcher
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont, QKeySequence, QShortcut
import cv2
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Optional, Dict, Any

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

# Robot imports
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
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


class MonitoringThread(QThread):
    """Thread for monitoring robot and camera."""
    
    camera_update = Signal(object)  # QPixmap
    motor_update = Signal(dict)  # Motor data dict
    status_update = Signal(str)  # Status message
    
    def __init__(self, bus, camera, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.camera = camera
        self.running = False
        
    def run(self):
        """Main monitoring loop."""
        try:
            # Connect systems
            self.bus._connect(handshake=False)
            self.camera.connect()
            
            self.status_update.emit("✅ Robot and camera connected - Live monitoring active")
            
            while self.running:
                # Update camera
                try:
                    if self.camera.is_connected:
                        frame = self.camera.read()
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (320, 240))
                        
                        height, width, channel = frame_resized.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_image)
                        self.camera_update.emit(pixmap)
                except Exception as e:
                    pass
                
                # Update motors - read all available signals
                try:
                    if self.bus.is_connected:
                        motor_data = {}
                        for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                                         "wrist_flex", "wrist_roll", "gripper"]:
                            try:
                                # Read all available signals
                                pos = self.bus.read("Present_Position", motor_name, normalize=True)
                                
                                # Try to read additional signals (may fail for some motors)
                                signals = {"position": pos, "status": "ok"}
                                
                                try:
                                    signals["velocity"] = self.bus.read("Present_Velocity", motor_name, normalize=False)
                                except:
                                    signals["velocity"] = None
                                
                                try:
                                    signals["load"] = self.bus.read("Present_Load", motor_name, normalize=False)
                                except:
                                    signals["load"] = None
                                
                                try:
                                    signals["voltage"] = self.bus.read("Present_Voltage", motor_name, normalize=False)
                                except:
                                    signals["voltage"] = None
                                
                                try:
                                    signals["temperature"] = self.bus.read("Present_Temperature", motor_name, normalize=False)
                                except:
                                    signals["temperature"] = None
                                
                                try:
                                    signals["current"] = self.bus.read("Present_Current", motor_name, normalize=False)
                                except:
                                    signals["current"] = None
                                
                                try:
                                    signals["moving"] = self.bus.read("Moving", motor_name, normalize=False)
                                except:
                                    signals["moving"] = None
                                
                                try:
                                    signals["goal_position"] = self.bus.read("Goal_Position", motor_name, normalize=True)
                                except:
                                    signals["goal_position"] = None
                                
                                try:
                                    signals["torque_enable"] = self.bus.read("Torque_Enable", motor_name, normalize=False)
                                except:
                                    signals["torque_enable"] = None
                                
                                motor_data[motor_name] = signals
                            except:
                                motor_data[motor_name] = {"position": None, "status": "error"}
                        self.motor_update.emit(motor_data)
                except Exception as e:
                    pass
                
                time.sleep(0.1)  # 10 FPS update rate
                
        except Exception as e:
            self.status_update.emit(f"❌ Monitoring error: {e}")
        finally:
            try:
                if hasattr(self, 'bus'):
                    self.bus.disconnect()
                if hasattr(self, 'camera'):
                    self.camera.disconnect()
            except:
                pass
    
    def stop(self):
        """Stop monitoring."""
        self.running = False


class ChessRobotUILLM(QMainWindow):
    def __init__(self, port: str, dev_mode: bool = False, api_key: Optional[str] = None):
        super().__init__()
        self.port = port
        self.running = False
        self.dev_mode = dev_mode
        self.file_watcher = None
        
        # Setup LLM
        self.setup_llm(api_key)
        
        # Setup robot and camera
        self.setup_robot()
        self.setup_camera()
        self.setup_kinematics()
        
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
    
    def setup_robot(self):
        """Initialize robot connection."""
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
        
    def setup_camera(self):
        """Initialize camera connection."""
        cfg = OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)
        self.camera = OpenCVCamera(cfg)
    
    def setup_kinematics(self):
        """Initialize robot kinematics for forward kinematics calculations."""
        try:
            # Check if URDF exists
            urdf_path = "./SO101/so101_new_calib.urdf"
            if not Path(urdf_path).exists():
                urdf_path = None
                print("⚠️ URDF not found - using simplified coordinate calculation")
            
            if urdf_path:
                joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
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
    
    def setup_llm(self, api_key: Optional[str] = None):
        """Initialize LLM client for natural language control."""
        self.llm_client = None
        self.llm_enabled = False
        
        if not LLM_AVAILABLE:
            print("⚠️ LLM support not available - install openai package")
            return
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if api_key:
            try:
                self.llm_client = OpenAI(api_key=api_key)
                self.llm_enabled = True
                print("✅ LLM client initialized")
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
        
        # Left column: Camera view
        self.create_camera_panel(panels_layout)
        
        # Right column: Motor status (more space now)
        self.create_motor_panel(panels_layout)
        
        # Give motor panel more space
        panels_layout.setStretch(0, 1)  # Camera
        panels_layout.setStretch(1, 2)  # Motors get 2x space
        
        main_layout.addLayout(panels_layout, 1)
        
        # Bottom row: 3D Robot visualization and LLM panel (give more space to LLM)
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(12)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.create_robot_3d_panel(bottom_layout)
        llm_widget = self.create_llm_panel(bottom_layout)
        # Give LLM panel more space (stretch factor of 2 vs 1 for 3D)
        bottom_layout.setStretch(0, 1)  # 3D panel
        bottom_layout.setStretch(1, 2)  # LLM panel gets 2x space
        main_layout.addLayout(bottom_layout, 1)
        
        # Bottom: Control buttons
        self.create_control_panel(main_layout)
        
    def create_camera_panel(self, parent_layout):
        """Create camera view panel."""
        camera_group = QGroupBox("Camera")
        # Terminal styling applied globally
        camera_group.setToolTip("Live camera feed")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(8)
        camera_layout.setContentsMargins(8, 8, 8, 8)
        
        # Camera display - terminal style
        self.camera_label = QLabel("Initializing...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(300, 220)
        self.camera_label.setMaximumSize(420, 320)
        self.camera_label.setScaledContents(True)
        camera_layout.addWidget(self.camera_label)
        
        # Camera info bar
        camera_info = QHBoxLayout()
        camera_info.setSpacing(10)
        
        self.fps_label = QLabel("FPS: --")
        camera_info.addWidget(self.fps_label)
        camera_info.addStretch()
        
        res_label = QLabel("640x480")
        camera_info.addWidget(res_label)
        camera_layout.addLayout(camera_info)
        
        # Camera status
        self.camera_status = QLabel("[CONNECTING]")
        self.camera_status.setStyleSheet("color: #f0883e;")
        self.camera_status.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.camera_status)
        
        parent_layout.addWidget(camera_group)
        
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
        """Create LLM natural language control panel."""
        llm_group = QGroupBox("LLM Control")
        # Terminal-like styling is applied globally, no need for custom styles here
        llm_group.setToolTip("Control robot using natural language commands. LLM generates motor positions from your instructions.")
        llm_layout = QVBoxLayout(llm_group)
        llm_layout.setSpacing(12)
        llm_layout.setContentsMargins(12, 12, 12, 12)
        
        # Model selection and status row
        model_status_layout = QHBoxLayout()
        model_status_layout.setSpacing(10)
        
        # Model selection
        model_label = QLabel("Model:")
        model_status_layout.addWidget(model_label)
        
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.addItems([
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ])
        self.llm_model_combo.setCurrentText("gpt-4o-mini")
        self.llm_model_combo.setMinimumWidth(150)
        model_status_layout.addWidget(self.llm_model_combo)
        model_status_layout.addStretch()
        
        # LLM status
        self.llm_status = QLabel("[READY]" if self.llm_enabled else "[NOT AVAILABLE]")
        if self.llm_enabled:
            self.llm_status.setStyleSheet("color: #3fb950;")  # Green for ready
        else:
            self.llm_status.setStyleSheet("color: #f85149;")  # Red for not available
        self.llm_status.setAlignment(Qt.AlignCenter)
        model_status_layout.addWidget(self.llm_status)
        llm_layout.addLayout(model_status_layout)
        
        # Command input
        command_label = QLabel("> Command:")
        llm_layout.addWidget(command_label)
        
        self.llm_command_input = QPlainTextEdit()
        self.llm_command_input.setPlaceholderText("Enter command... (e.g., move arm to e2)")
        self.llm_command_input.setMinimumHeight(80)
        llm_layout.addWidget(self.llm_command_input)
        
        # Execute button
        execute_btn = QPushButton("EXECUTE")
        execute_btn.setEnabled(self.llm_enabled)
        execute_btn.clicked.connect(self.execute_llm_command)
        llm_layout.addWidget(execute_btn)
        
        # LLM Reasoning/Explanation display (larger, more prominent)
        reasoning_label = QLabel("> Reasoning:")
        llm_layout.addWidget(reasoning_label)
        
        self.llm_reasoning_display = QTextEdit()
        self.llm_reasoning_display.setReadOnly(True)
        self.llm_reasoning_display.setPlaceholderText("LLM reasoning will appear here...")
        self.llm_reasoning_display.setStyleSheet("color: #f0883e;")  # Orange for reasoning
        self.llm_reasoning_display.setMinimumHeight(150)
        llm_layout.addWidget(self.llm_reasoning_display, 2)  # Give it stretch factor of 2
        
        # LLM response display (full JSON)
        response_label = QLabel("> Response (JSON):")
        llm_layout.addWidget(response_label)
        
        self.llm_response_display = QTextEdit()
        self.llm_response_display.setReadOnly(True)
        self.llm_response_display.setStyleSheet("color: #8b949e;")  # Gray for JSON
        self.llm_response_display.setMinimumHeight(120)
        llm_layout.addWidget(self.llm_response_display, 1)
        
        # Action preview
        preview_label = QLabel("> Action (Validated):")
        llm_layout.addWidget(preview_label)
        
        self.llm_action_preview = QTextEdit()
        self.llm_action_preview.setReadOnly(True)
        self.llm_action_preview.setStyleSheet("color: #58a6ff;")  # Blue for action
        self.llm_action_preview.setMinimumHeight(100)
        llm_layout.addWidget(self.llm_action_preview, 1)
        
        parent_layout.addWidget(llm_group)
        return llm_group  # Return widget for stretch factor setting
    
    def execute_llm_command(self):
        """Execute LLM command and control robot."""
        if not self.llm_enabled or not self.llm_client:
            self.status_bar.setText("❌ LLM not available")
            return
        
        command = self.llm_command_input.toPlainText().strip()
        if not command:
            self.status_bar.setText("⚠️ Please enter a command")
            return
        
        self.status_bar.setText("🤖 Processing LLM command...")
        self.llm_response_display.setText("Processing...")
        self.llm_reasoning_display.setText("Processing...")
        self.llm_action_preview.setText("")
        
        try:
            # Get current robot state
            current_positions = {}
            for motor_name in self.all_motors.keys():
                try:
                    pos = self.bus.read("Present_Position", motor_name, normalize=True)
                    current_positions[motor_name] = pos
                except:
                    current_positions[motor_name] = 0
            
            # Get selected model
            selected_model = self.llm_model_combo.currentText()
            
            # Build prompt for LLM
            prompt = self._build_llm_prompt(command, current_positions)
            
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You are a robot control assistant. Provide detailed reasoning for your actions. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3  # Lower temperature for more consistent outputs
            )
            
            # Parse response
            llm_output = json.loads(response.choices[0].message.content)
            
            # Display full response
            self.llm_response_display.setText(json.dumps(llm_output, indent=2))
            
            # Display reasoning/explanation prominently
            explanation = llm_output.get("explanation", "No explanation provided")
            reasoning_text = f"{explanation}\n\n"
            
            # Add reasoning about motor changes if available
            action = llm_output.get("action", {})
            if action:
                reasoning_text += "Motor Changes:\n"
                for motor_key, value in action.items():
                    motor_name = motor_key.replace(".pos", "")
                    current = current_positions.get(motor_name, 0)
                    change = value - current
                    reasoning_text += f"  • {motor_name}: {current:.1f}° → {value:.1f}° (Δ{change:+.1f}°)\n"
            
            self.llm_reasoning_display.setText(reasoning_text)
            
            # Validate and execute
            safe_action = self._validate_llm_action(action, current_positions)
            
            if safe_action:
                self.llm_action_preview.setText(json.dumps(safe_action, indent=2))
                self._execute_llm_action(safe_action)
                self.status_bar.setText(f"✅ LLM command executed successfully (using {selected_model})")
            else:
                self.status_bar.setText("❌ LLM action failed validation")
                self.llm_action_preview.setText("Action validation failed - unsafe values detected")
                self.llm_reasoning_display.setText(
                    self.llm_reasoning_display.toPlainText() + 
                    "\n\n⚠️ VALIDATION FAILED: Action was rejected for safety reasons."
                )
                
        except json.JSONDecodeError as e:
            self.status_bar.setText(f"❌ Invalid JSON from LLM: {e}")
            self.llm_response_display.setText(f"Error: Invalid JSON response\n{response.choices[0].message.content if 'response' in locals() else 'No response'}")
        except Exception as e:
            self.status_bar.setText(f"❌ LLM error: {str(e)[:50]}...")
            self.llm_response_display.setText(f"Error: {str(e)}")
    
    def _build_llm_prompt(self, command: str, current_positions: Dict[str, float]) -> str:
        """Build prompt for LLM with robot context."""
        prompt = f"""You control a 5-DOF robot arm for chess piece manipulation.

Available motors:
- shoulder_pan: horizontal rotation (degrees, range: -180 to 180)
- shoulder_lift: vertical lift (degrees, range: -90 to 90)
- elbow_flex: elbow bend (degrees, range: -90 to 90)
- wrist_flex: wrist bend (degrees, range: -90 to 90)
- gripper: open/close (0-100%, 0=open, 100=closed)

Current motor positions:
{json.dumps(current_positions, indent=2)}

User command: {command}

Output a JSON object with this structure:
{{
  "action": {{
    "shoulder_pan.pos": <float>,
    "shoulder_lift.pos": <float>,
    "elbow_flex.pos": <float>,
    "wrist_flex.pos": <float>,
    "gripper.pos": <float>
  }},
  "explanation": "<brief explanation of the action>"
}}

Important:
- Only include motors that need to change
- Keep changes small and safe (max 10-20 degrees per step)
- Ensure gripper values are between 0-100
- Ensure joint angles are within safe ranges
"""
        return prompt
    
    def _validate_llm_action(self, action: Dict[str, Any], current_positions: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Validate and clamp LLM-generated action to safe values."""
        if not isinstance(action, dict):
            return None
        
        safe_action = {}
        max_change = 20.0  # Maximum degrees change per step
        
        for motor_name, value in action.items():
            # Remove .pos suffix if present
            motor_key = motor_name.replace(".pos", "")
            
            if motor_key not in self.all_motors:
                continue
            
            try:
                target_value = float(value)
                current_value = current_positions.get(motor_key, 0)
                
                # Check change magnitude
                change = abs(target_value - current_value)
                if change > max_change:
                    # Clamp to max change
                    if target_value > current_value:
                        target_value = current_value + max_change
                    else:
                        target_value = current_value - max_change
                
                # Gripper-specific validation
                if motor_key == "gripper":
                    target_value = max(0.0, min(100.0, target_value))
                else:
                    # Joint angle validation (rough bounds)
                    target_value = max(-180.0, min(180.0, target_value))
                
                safe_action[f"{motor_key}.pos"] = target_value
                
            except (ValueError, TypeError):
                continue
        
        return safe_action if safe_action else None
    
    def _execute_llm_action(self, action: Dict[str, float]):
        """Execute validated LLM action on robot."""
        try:
            for motor_key, value in action.items():
                motor_name = motor_key.replace(".pos", "")
                if motor_name in self.all_motors:
                    self.bus.write("Goal_Position", motor_name, value, normalize=True)
                    time.sleep(0.1)  # Small delay between commands
            
            self.status_bar.setText("✅ LLM action executed")
        except Exception as e:
            self.status_bar.setText(f"❌ Execution error: {str(e)[:50]}...")
            raise
    
    def move_gripper(self, dx_mm, dy_mm, dz_mm):
        """Move gripper by specified amounts in base coordinates."""
        try:
            step_size = float(self.step_size_combo.currentText())
            dx = dx_mm * step_size / 10.0
            dy = dy_mm * step_size / 10.0
            dz = dz_mm * step_size / 10.0
            
            self.status_bar.setText(f"🎯 Moving gripper: Δx={dx:.1f}, Δy={dy:.1f}, Δz={dz:.1f} mm")
            
            # Get current joint positions
            current_joints = {}
            for motor_name in self.all_motors.keys():
                if motor_name != "gripper":
                    pos = self.bus.read("Present_Position", motor_name, normalize=True)
                    current_joints[motor_name] = pos
            
            # Get current end-effector position
            current_x, current_y, current_z, _ = self.calculate_base_coordinates(current_joints)
            
            # Calculate target position
            target_x = current_x + dx
            target_y = current_y + dy  
            target_z = current_z + dz
            
            # Check workspace bounds
            distance = np.sqrt(target_x**2 + target_y**2 + target_z**2)
            if distance > 450:
                self.status_bar.setText("❌ Target outside workspace bounds!")
                return
            
            if distance < 80:
                self.status_bar.setText("❌ Target too close to base!")
                return
                
            # Use inverse kinematics if available
            if self.kinematics is not None:
                try:
                    # Create target transformation matrix
                    T_target = np.eye(4)
                    T_target[0, 3] = target_x / 1000.0
                    T_target[1, 3] = target_y / 1000.0
                    T_target[2, 3] = target_z / 1000.0
                    
                    # Calculate required joint angles
                    current_joint_array = [current_joints[name] for name in 
                                         ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]]
                    
                    target_joints = self.kinematics.inverse_kinematics(current_joint_array, T_target)
                    
                    # Execute movement with small increments
                    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
                    for i, motor_name in enumerate(joint_names):
                        if i < len(target_joints):
                            self.bus.write("Goal_Position", motor_name, float(target_joints[i]))
                            time.sleep(0.2)
                    
                    self.status_bar.setText(f"✅ Moved to ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
                    
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
            self.status_bar.setText(f"❌ Movement failed: {str(e)[:50]}...")
    
    def update_workspace_position(self, x_mm, y_mm, z_mm):
        """Update robot position on workspace visualization."""
        self.workspace_canvas.set_robot_position(x_mm, y_mm, z_mm)
    
    def update_workspace_display(self, x_mm, y_mm, z_mm):
        """Update workspace visualization with current robot position."""
        self.workspace_side_canvas.set_robot_position(x_mm, y_mm, z_mm)
        self.workspace_top_canvas.set_robot_position(x_mm, y_mm, z_mm)
    
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
    
    def update_camera(self, pixmap):
        """Update camera view from monitoring thread."""
        self.camera_label.setPixmap(pixmap)
        self.camera_status.setText("[LIVE]")
        self.camera_status.setStyleSheet("color: #3fb950;")
        
        # Update FPS (simple counter - could be improved)
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
            self._last_fps_time = time.time()
        self._frame_count += 1
        current_time = time.time()
        if current_time - self._last_fps_time >= 1.0:
            fps = self._frame_count / (current_time - self._last_fps_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._frame_count = 0
            self._last_fps_time = current_time
    
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
        self.monitor_thread = MonitoringThread(self.bus, self.camera)
        self.monitor_thread.camera_update.connect(self.update_camera)
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
    
    def refresh_status(self):
        """Force refresh of all displays."""
        self.status_bar.setText("🔄 Refreshing all systems...")
        # The monitoring thread will update automatically
        QTimer.singleShot(1000, lambda: self.status_bar.setText("✅ Refresh complete"))
    
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


def main():
    import argparse
    p = argparse.ArgumentParser(description="Chess Robot Monitoring UI with LLM Control")
    p.add_argument("--port", required=True, help="Robot serial port")
    p.add_argument("--dev", action="store_true", help="Enable development mode with file watching")
    p.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = p.parse_args()
    
    app = QApplication([])
    
    try:
        ui = ChessRobotUILLM(args.port, dev_mode=args.dev, api_key=args.api_key)
        ui.show()
        print("🚀 Starting Chess Robot UI with LLM Control...")
        if args.dev:
            print("🔧 Development mode: File watching enabled")
        if ui.llm_enabled:
            print("✅ LLM control enabled")
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
