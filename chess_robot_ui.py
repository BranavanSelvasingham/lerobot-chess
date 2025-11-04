#!/usr/bin/env python

"""
Chess Robot Monitoring UI
Shows live camera feed, motor status, and chess board diagram.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk

# Robot imports
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.model.kinematics import RobotKinematics

class ChessRobotUI:
    def __init__(self, port: str):
        self.port = port
        self.running = False
        
        # Setup robot and camera
        self.setup_robot()
        self.setup_camera()
        self.setup_kinematics()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Chess Robot Monitor")
        self.root.geometry("1600x900")
        self.root.configure(bg='#2c3e50')
        
        # Initialize UI components
        self.create_widgets()
        
        # Start monitoring threads
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
        
    def create_widgets(self):
        """Create all UI widgets."""
        
        # Main title
        title = tk.Label(self.root, text="♟️ Chess Robot Monitor 🤖", 
                        font=("Arial", 20, "bold"), 
                        bg='#2c3e50', fg='#ecf0f1')
        title.pack(pady=10)
        
        # Create main frame with three columns
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left column: Camera view
        self.create_camera_panel(main_frame)
        
        # Middle column: Motor status
        self.create_motor_panel(main_frame)
        
        # Right column: Chess board and coordinates
        self.create_chess_panel(main_frame)
        
        # Far right: Robot base coordinates and workspace
        self.create_coordinates_panel(main_frame)
        
        # Bottom row: Workspace control and visualization
        self.create_workspace_control_panel()
        
        # Bottom: Control buttons
        self.create_control_panel()
        
    def create_camera_panel(self, parent):
        """Create camera view panel."""
        camera_frame = tk.LabelFrame(parent, text="📸 Robot Camera View", 
                                   font=("Arial", 12, "bold"),
                                   bg='#34495e', fg='#ecf0f1')
        camera_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Camera display
        self.camera_label = tk.Label(camera_frame, text="Initializing camera...", 
                                   bg='#2c3e50', fg='#95a5a6')
        self.camera_label.pack(pady=10)
        
        # Camera status
        self.camera_status = tk.Label(camera_frame, text="Status: Connecting", 
                                    font=("Arial", 10),
                                    bg='#34495e', fg='#f39c12')
        self.camera_status.pack()
        
    def create_motor_panel(self, parent):
        """Create motor status panel."""
        motor_frame = tk.LabelFrame(parent, text="🤖 Motor Status", 
                                  font=("Arial", 12, "bold"),
                                  bg='#34495e', fg='#ecf0f1')
        motor_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        # Motor status displays
        self.motor_labels = {}
        self.motor_status_labels = {}
        
        for i, motor_name in enumerate(self.all_motors.keys()):
            # Motor name and position
            motor_row = tk.Frame(motor_frame, bg='#34495e')
            motor_row.pack(fill=tk.X, padx=5, pady=2)
            
            name_label = tk.Label(motor_row, text=f"{motor_name}:", 
                                font=("Arial", 10, "bold"),
                                bg='#34495e', fg='#ecf0f1', width=12, anchor='w')
            name_label.pack(side=tk.LEFT)
            
            pos_label = tk.Label(motor_row, text="--°", 
                               font=("Arial", 10, "bold"),
                               bg='#34495e', fg='#3498db', width=8)
            pos_label.pack(side=tk.LEFT, padx=5)
            
            status_label = tk.Label(motor_row, text="●", 
                                  font=("Arial", 12),
                                  bg='#34495e', fg='#95a5a6')
            status_label.pack(side=tk.RIGHT)
            
            self.motor_labels[motor_name] = pos_label
            self.motor_status_labels[motor_name] = status_label
        
        # Overall robot status
        self.robot_status = tk.Label(motor_frame, text="Robot: Connecting...", 
                                   font=("Arial", 10, "bold"),
                                   bg='#34495e', fg='#f39c12')
        self.robot_status.pack(pady=10)
        
    def create_chess_panel(self, parent):
        """Create chess board visualization panel."""
        chess_frame = tk.LabelFrame(parent, text="♟️ Chess Board", 
                                  font=("Arial", 12, "bold"),
                                  bg='#34495e', fg='#ecf0f1')
        chess_frame.grid(row=0, column=2, padx=10, pady=5, sticky="nsew")
        
        # Chess board canvas (larger for better visibility)
        self.chess_canvas = tk.Canvas(chess_frame, width=260, height=260, 
                                    bg='#2c3e50', highlightthickness=0)
        self.chess_canvas.pack(pady=10)
        
        # Robot position details
        self.robot_details = tk.Label(chess_frame, text="Position: --\nAccuracy: --", 
                                    font=("Arial", 9),
                                    bg='#34495e', fg='#bdc3c7', justify=tk.LEFT)
        self.robot_details.pack(pady=5)
        
        # Robot position indicator
        self.robot_position_label = tk.Label(chess_frame, text="Robot at: --", 
                                           font=("Arial", 10),
                                           bg='#34495e', fg='#e74c3c')
        self.robot_position_label.pack()
    
    def create_coordinates_panel(self, parent):
        """Create robot base coordinates panel."""
        coord_frame = tk.LabelFrame(parent, text="🔧 Robot Base Coordinates", 
                                  font=("Arial", 12, "bold"),
                                  bg='#34495e', fg='#ecf0f1')
        coord_frame.grid(row=0, column=3, padx=10, pady=5, sticky="nsew")
        
        # End-effector position in base frame
        self.ee_position_label = tk.Label(coord_frame, text="End-Effector Position:", 
                                        font=("Arial", 10, "bold"),
                                        bg='#34495e', fg='#ecf0f1')
        self.ee_position_label.pack(pady=5)
        
        # X, Y, Z coordinates
        self.coord_x = tk.Label(coord_frame, text="X: ---.-- mm", 
                              font=("Arial", 11, "bold"),
                              bg='#34495e', fg='#e74c3c')
        self.coord_x.pack(pady=2)
        
        self.coord_y = tk.Label(coord_frame, text="Y: ---.-- mm", 
                              font=("Arial", 11, "bold"),
                              bg='#34495e', fg='#27ae60')
        self.coord_y.pack(pady=2)
        
        self.coord_z = tk.Label(coord_frame, text="Z: ---.-- mm", 
                              font=("Arial", 11, "bold"),
                              bg='#34495e', fg='#3498db')
        self.coord_z.pack(pady=2)
        
        # Workspace info
        tk.Label(coord_frame, text="Workspace Status:", 
               font=("Arial", 10, "bold"),
               bg='#34495e', fg='#ecf0f1').pack(pady=(10,5))
        
        self.workspace_status = tk.Label(coord_frame, text="Position: --\nReach: --", 
                                       font=("Arial", 9),
                                       bg='#34495e', fg='#bdc3c7', justify=tk.LEFT)
        self.workspace_status.pack(pady=2)
        
        # Distance from base
        self.distance_label = tk.Label(coord_frame, text="Distance: --- mm", 
                                     font=("Arial", 9, "bold"),
                                     bg='#34495e', fg='#f39c12')
        self.distance_label.pack(pady=5)
        
        # Joint configuration display
        tk.Label(coord_frame, text="Joint Configuration:", 
               font=("Arial", 10, "bold"),
               bg='#34495e', fg='#ecf0f1').pack(pady=(10,5))
        
        self.joint_config = tk.Label(coord_frame, 
                                   text="Shoulder: --°, --°\nElbow: --°\nWrist: --°, --°", 
                                   font=("Arial", 8),
                                   bg='#34495e', fg='#95a5a6', justify=tk.LEFT)
        self.joint_config.pack()
        
        # Simple 2D workspace visualization
        tk.Label(coord_frame, text="Top View:", 
               font=("Arial", 10, "bold"),
               bg='#34495e', fg='#ecf0f1').pack(pady=(10,5))
        
        self.workspace_canvas = tk.Canvas(coord_frame, width=120, height=120, 
                                        bg='#2c3e50', highlightthickness=0)
        self.workspace_canvas.pack()
        
        # Draw workspace
        self.draw_workspace()
        
    def create_control_panel(self):
        """Create control buttons panel."""
        control_frame = tk.Frame(self.root, bg='#2c3e50')
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Control buttons
        tk.Button(control_frame, text="🏠 Go Home", command=self.go_home,
                 font=("Arial", 10, "bold"), bg='#27ae60', fg='white',
                 relief=tk.FLAT, padx=20).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="🔄 Refresh", command=self.refresh_status,
                 font=("Arial", 10, "bold"), bg='#3498db', fg='white',
                 relief=tk.FLAT, padx=20).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="🛑 Stop", command=self.stop_monitoring,
                 font=("Arial", 10, "bold"), bg='#e74c3c', fg='white',
                 relief=tk.FLAT, padx=20).pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_bar = tk.Label(control_frame, text="Initializing...", 
                                 font=("Arial", 9),
                                 bg='#2c3e50', fg='#95a5a6')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
    def draw_chess_board(self):
        """Draw enhanced chess board diagram with better visualization."""
        # Clear previous drawings but preserve square labels
        self.chess_canvas.delete("robot_pos")
        
        # Only redraw board if not already drawn
        if not hasattr(self, '_board_drawn'):
            self.chess_canvas.delete("all")
            
            # Draw 8x8 grid with enhanced styling
            square_size = 30
            for row in range(8):
                for col in range(8):
                    x1 = col * square_size
                    y1 = row * square_size
                    x2 = x1 + square_size
                    y2 = y1 + square_size
                    
                    # Alternating colors with better contrast
                    if (row + col) % 2 == 0:
                        color = '#f8f9fa'  # Light squares (almost white)
                    else:
                        color = '#6c757d'  # Dark squares (gray)
                    
                    self.chess_canvas.create_rectangle(x1, y1, x2, y2, 
                                                     fill=color, outline='#343a40', width=1,
                                                     tags="board")
                    
                    # Add square labels with better visibility
                    square = chr(ord('a') + col) + str(8 - row)
                    text_color = '#343a40' if (row + col) % 2 == 0 else '#f8f9fa'
                    self.chess_canvas.create_text(x1 + 15, y1 + 8, text=square,
                                                font=("Arial", 7, "bold"), fill=text_color,
                                                tags="board")
            
            # Add board edge labels
            # File labels (a-h) at bottom
            for col in range(8):
                x = col * square_size + 15
                self.chess_canvas.create_text(x, 245, text=chr(ord('a') + col),
                                            font=("Arial", 10, "bold"), fill='#ecf0f1')
            
            # Rank labels (1-8) on right side
            for row in range(8):
                y = row * square_size + 15
                self.chess_canvas.create_text(245, y, text=str(8 - row),
                                            font=("Arial", 10, "bold"), fill='#ecf0f1')
            
            self._board_drawn = True
    
    def draw_workspace(self):
        """Draw 2D workspace visualization (top view)."""
        self.workspace_canvas.delete("all")
        
        # Draw robot base (center)
        center = 60
        self.workspace_canvas.create_oval(center-3, center-3, center+3, center+3,
                                        fill='#2c3e50', outline='#ecf0f1', width=2)
        self.workspace_canvas.create_text(center, center-10, text="BASE",
                                        font=("Arial", 6), fill='#ecf0f1')
        
        # Draw workspace boundary (approximate)
        self.workspace_canvas.create_oval(10, 10, 110, 110,
                                        outline='#95a5a6', width=1, dash=(2,2))
        
        # Draw coordinate axes
        # X-axis (forward)
        self.workspace_canvas.create_line(center, center, center, 20,
                                        fill='#e74c3c', width=2, arrow=tk.LAST)
        self.workspace_canvas.create_text(center+10, 25, text="X", 
                                        font=("Arial", 8, "bold"), fill='#e74c3c')
        
        # Y-axis (right)  
        self.workspace_canvas.create_line(center, center, 100, center,
                                        fill='#27ae60', width=2, arrow=tk.LAST)
        self.workspace_canvas.create_text(95, center-10, text="Y",
                                        font=("Arial", 8, "bold"), fill='#27ae60')
    
    def update_workspace_position(self, x_mm, y_mm, z_mm):
        """Update robot position on workspace visualization."""
        # Clear previous position
        self.workspace_canvas.delete("robot_pos")
        
        # Scale coordinates to canvas (assuming ~400mm max reach)
        center = 60
        scale = 50 / 400  # 50 pixels = 400mm
        
        # Convert to canvas coordinates
        canvas_x = center + y_mm * scale  # Y maps to canvas X (right)
        canvas_y = center - x_mm * scale  # X maps to canvas Y (up), inverted
        
        # Ensure within canvas bounds
        canvas_x = max(10, min(110, canvas_x))
        canvas_y = max(10, min(110, canvas_y))
        
        # Draw robot end-effector position
        self.workspace_canvas.create_oval(canvas_x-4, canvas_y-4, canvas_x+4, canvas_y+4,
                                        fill='#e74c3c', outline='#ffffff', width=2,
                                        tags="robot_pos")
        
        # Add height indicator (Z-axis as circle size/color)
        if z_mm > 0:
            color_intensity = min(255, int(z_mm / 200 * 255))  # Scale with height
            height_color = f"#{color_intensity:02x}{color_intensity:02x}ff"
            
            self.workspace_canvas.create_oval(canvas_x-6, canvas_y-6, canvas_x+6, canvas_y+6,
                                            outline=height_color, width=1,
                                            tags="robot_pos")
    
    def calculate_base_coordinates(self, joint_positions):
        """Calculate end-effector position in robot base frame."""
        try:
            if self.kinematics is not None:
                # Use full forward kinematics
                joint_angles = [joint_positions[name] for name in 
                              ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]]
                
                T_base_ee = self.kinematics.forward_kinematics(joint_angles)
                
                # Extract position from transformation matrix
                x_mm = T_base_ee[0, 3] * 1000  # Convert to mm
                y_mm = T_base_ee[1, 3] * 1000
                z_mm = T_base_ee[2, 3] * 1000
                
                return x_mm, y_mm, z_mm, "Full FK"
                
            else:
                # Simplified approximation using joint angles
                # This is a rough approximation for visualization
                shoulder_pan = np.radians(joint_positions["shoulder_pan"])
                shoulder_lift = np.radians(joint_positions["shoulder_lift"]) 
                elbow_flex = np.radians(joint_positions["elbow_flex"])
                wrist_flex = np.radians(joint_positions["wrist_flex"])
                
                # Approximate arm segment lengths (mm)
                L1 = 150  # Shoulder to elbow
                L2 = 120  # Elbow to wrist  
                L3 = 80   # Wrist to gripper
                
                # Forward kinematics approximation
                # Shoulder lift + elbow angle
                total_lift = shoulder_lift + elbow_flex
                
                # Calculate reach
                reach = L1 * np.cos(shoulder_lift) + L2 * np.cos(total_lift) + L3 * np.cos(total_lift + wrist_flex)
                height = L1 * np.sin(shoulder_lift) + L2 * np.sin(total_lift) + L3 * np.sin(total_lift + wrist_flex)
                
                # Apply shoulder pan rotation
                x_mm = reach * np.cos(shoulder_pan)
                y_mm = reach * np.sin(shoulder_pan)
                z_mm = height + 200  # Base height offset
                
                return x_mm, y_mm, z_mm, "Approximation"
                
        except Exception as e:
            return 0, 0, 0, f"Error: {e}"
    
    def create_workspace_control_panel(self):
        """Create workspace visualization and control panel."""
        workspace_frame = tk.LabelFrame(self.root, text="🎯 Workspace Control & Visualization", 
                                      font=("Arial", 12, "bold"),
                                      bg='#34495e', fg='#ecf0f1')
        workspace_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Split into visualization and controls
        viz_frame = tk.Frame(workspace_frame, bg='#34495e')
        viz_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        control_frame = tk.Frame(workspace_frame, bg='#34495e')
        control_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Workspace visualization (side view and top view)
        tk.Label(viz_frame, text="Workspace Volume (Side View)", 
               font=("Arial", 10, "bold"), bg='#34495e', fg='#ecf0f1').pack()
        
        self.workspace_side_canvas = tk.Canvas(viz_frame, width=300, height=200, 
                                            bg='#2c3e50', highlightthickness=1,
                                            highlightbackground='#95a5a6')
        self.workspace_side_canvas.pack(pady=5)
        
        tk.Label(viz_frame, text="Workspace Volume (Top View)", 
               font=("Arial", 10, "bold"), bg='#34495e', fg='#ecf0f1').pack()
        
        self.workspace_top_canvas = tk.Canvas(viz_frame, width=300, height=200, 
                                           bg='#2c3e50', highlightthickness=1,
                                           highlightbackground='#95a5a6')
        self.workspace_top_canvas.pack(pady=5)
        
        # Interactive controls
        tk.Label(control_frame, text="Interactive Gripper Control", 
               font=("Arial", 12, "bold"), bg='#34495e', fg='#ecf0f1').pack(pady=5)
        
        # Current position display
        self.current_pos_label = tk.Label(control_frame, text="Current Position:\nX: --- mm\nY: --- mm\nZ: --- mm", 
                                        font=("Arial", 10), bg='#34495e', fg='#bdc3c7',
                                        justify=tk.LEFT)
        self.current_pos_label.pack(pady=5)
        
        # Movement controls
        move_frame = tk.LabelFrame(control_frame, text="Move Gripper", bg='#34495e', fg='#ecf0f1')
        move_frame.pack(pady=10, padx=5, fill=tk.X)
        
        # X-axis controls (forward/back)
        x_frame = tk.Frame(move_frame, bg='#34495e')
        x_frame.pack(fill=tk.X, pady=2)
        tk.Label(x_frame, text="X:", font=("Arial", 10, "bold"), bg='#34495e', fg='#e74c3c').pack(side=tk.LEFT)
        tk.Button(x_frame, text="←Back", command=lambda: self.move_gripper(-10, 0, 0),
                 bg='#e74c3c', fg='white', font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(x_frame, text="Fwd→", command=lambda: self.move_gripper(10, 0, 0),
                 bg='#e74c3c', fg='white', font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        
        # Y-axis controls (left/right)
        y_frame = tk.Frame(move_frame, bg='#34495e')
        y_frame.pack(fill=tk.X, pady=2)
        tk.Label(y_frame, text="Y:", font=("Arial", 10, "bold"), bg='#34495e', fg='#27ae60').pack(side=tk.LEFT)
        tk.Button(y_frame, text="←Left", command=lambda: self.move_gripper(0, -10, 0),
                 bg='#27ae60', fg='white', font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(y_frame, text="Right→", command=lambda: self.move_gripper(0, 10, 0),
                 bg='#27ae60', fg='white', font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        
        # Z-axis controls (up/down)
        z_frame = tk.Frame(move_frame, bg='#34495e')
        z_frame.pack(fill=tk.X, pady=2)
        tk.Label(z_frame, text="Z:", font=("Arial", 10, "bold"), bg='#34495e', fg='#3498db').pack(side=tk.LEFT)
        tk.Button(z_frame, text="↓Down", command=lambda: self.move_gripper(0, 0, -10),
                 bg='#3498db', fg='white', font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(z_frame, text="Up↑", command=lambda: self.move_gripper(0, 0, 10),
                 bg='#3498db', fg='white', font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        
        # Movement size controls
        size_frame = tk.Frame(move_frame, bg='#34495e')
        size_frame.pack(fill=tk.X, pady=5)
        tk.Label(size_frame, text="Step Size:", font=("Arial", 9), bg='#34495e', fg='#ecf0f1').pack(side=tk.LEFT)
        
        self.step_size = tk.StringVar(value="10")
        step_sizes = ["5", "10", "20", "50"]
        step_combo = ttk.Combobox(size_frame, textvariable=self.step_size, values=step_sizes, width=8)
        step_combo.pack(side=tk.LEFT, padx=5)
        
        # Workspace bounds display
        bounds_frame = tk.LabelFrame(control_frame, text="Workspace Bounds", bg='#34495e', fg='#ecf0f1')
        bounds_frame.pack(pady=10, padx=5, fill=tk.X)
        
        self.workspace_bounds = tk.Label(bounds_frame, 
                                       text="X: [---,---] mm\nY: [---,---] mm\nZ: [---,---] mm", 
                                       font=("Arial", 9), bg='#34495e', fg='#95a5a6',
                                       justify=tk.LEFT)
        self.workspace_bounds.pack(pady=5)
        
        # Draw initial workspace
        self.draw_workspace_volume()
    
    def draw_workspace_volume(self):
        """Draw the robot's reachable workspace volume."""
        
        # Side view (X-Z plane)
        self.workspace_side_canvas.delete("all")
        
        # Draw coordinate system for side view
        center_x, center_z = 150, 100
        
        # Base position
        self.workspace_side_canvas.create_oval(center_x-3, 180, center_x+3, 186,
                                             fill='#2c3e50', outline='#ecf0f1', width=2)
        self.workspace_side_canvas.create_text(center_x, 195, text="BASE", 
                                             font=("Arial", 8), fill='#ecf0f1')
        
        # Draw approximate workspace envelope (side view)
        # This represents the reachable space based on arm geometry
        workspace_points_side = []
        
        # Calculate workspace boundary points
        max_reach = 350  # mm (approximate)
        min_reach = 100  # mm 
        
        # Create workspace envelope
        for angle in np.linspace(-np.pi/2, np.pi/2, 20):  # -90° to +90°
            # Outer boundary
            x_outer = center_x + max_reach * np.cos(angle) * 0.3  # Scale for display
            z_outer = center_z - max_reach * np.sin(angle) * 0.3
            workspace_points_side.extend([x_outer, z_outer])
            
        # Draw workspace envelope
        if len(workspace_points_side) >= 6:
            self.workspace_side_canvas.create_polygon(workspace_points_side,
                                                    outline='#3498db', width=2,
                                                    fill='#3498db', stipple='gray25')
        
        # Draw axes for side view
        self.workspace_side_canvas.create_line(center_x, 180, center_x + 60, 180,
                                             fill='#e74c3c', width=2, arrow=tk.LAST)
        self.workspace_side_canvas.create_text(center_x + 70, 180, text="X (forward)",
                                             font=("Arial", 8), fill='#e74c3c')
        
        self.workspace_side_canvas.create_line(center_x, 180, center_x, 120,
                                             fill='#3498db', width=2, arrow=tk.LAST)
        self.workspace_side_canvas.create_text(center_x + 15, 110, text="Z (up)",
                                             font=("Arial", 8), fill='#3498db')
        
        # Top view (X-Y plane)
        self.workspace_top_canvas.delete("all")
        
        center_top_x, center_top_y = 150, 100
        
        # Base position
        self.workspace_top_canvas.create_oval(center_top_x-3, center_top_y-3, 
                                            center_top_x+3, center_top_y+3,
                                            fill='#2c3e50', outline='#ecf0f1', width=2)
        self.workspace_top_canvas.create_text(center_top_x, center_top_y-15, text="BASE",
                                           font=("Arial", 8), fill='#ecf0f1')
        
        # Draw workspace circle (top view)
        scale = 0.3  # mm to pixels
        outer_radius = max_reach * scale
        inner_radius = min_reach * scale
        
        # Outer workspace boundary
        self.workspace_top_canvas.create_oval(center_top_x - outer_radius, center_top_y - outer_radius,
                                            center_top_x + outer_radius, center_top_y + outer_radius,
                                            outline='#3498db', width=2, dash=(5,5))
        
        # Inner dead zone
        self.workspace_top_canvas.create_oval(center_top_x - inner_radius, center_top_y - inner_radius,
                                            center_top_x + inner_radius, center_top_y + inner_radius,
                                            outline='#95a5a6', width=1, dash=(2,2))
        
        # Optimal workspace (middle zone)
        mid_radius = (max_reach + min_reach) / 2 * scale
        self.workspace_top_canvas.create_oval(center_top_x - mid_radius, center_top_y - mid_radius,
                                            center_top_x + mid_radius, center_top_y + mid_radius,
                                            outline='#27ae60', width=2)
        
        # Draw axes for top view
        self.workspace_top_canvas.create_line(center_top_x, center_top_y, center_top_x, center_top_y - 60,
                                            fill='#e74c3c', width=2, arrow=tk.LAST)
        self.workspace_top_canvas.create_text(center_top_x + 20, center_top_y - 50, text="X",
                                           font=("Arial", 8, "bold"), fill='#e74c3c')
        
        self.workspace_top_canvas.create_line(center_top_x, center_top_y, center_top_x + 60, center_top_y,
                                           fill='#27ae60', width=2, arrow=tk.LAST)
        self.workspace_top_canvas.create_text(center_top_x + 50, center_top_y - 15, text="Y",
                                           font=("Arial", 8, "bold"), fill='#27ae60')
        
        # Add workspace legend
        legend_y = 20
        self.workspace_top_canvas.create_text(20, legend_y, text="🟢 Optimal", 
                                            font=("Arial", 8), fill='#27ae60', anchor="w")
        self.workspace_top_canvas.create_text(20, legend_y + 15, text="🔵 Reachable", 
                                            font=("Arial", 8), fill='#3498db', anchor="w")
        self.workspace_top_canvas.create_text(20, legend_y + 30, text="⚪ Dead zone", 
                                            font=("Arial", 8), fill='#95a5a6', anchor="w")
    
    def move_gripper(self, dx_mm, dy_mm, dz_mm):
        """Move gripper by specified amounts in base coordinates."""
        try:
            step_size = float(self.step_size.get())
            dx = dx_mm * step_size / 10.0  # Scale by step size
            dy = dy_mm * step_size / 10.0
            dz = dz_mm * step_size / 10.0
            
            self.status_bar.configure(text=f"🎯 Moving gripper: Δx={dx:.1f}, Δy={dy:.1f}, Δz={dz:.1f} mm")
            
            # Get current joint positions
            current_joints = {}
            for motor_name in self.all_motors.keys():
                if motor_name != "gripper":  # Skip gripper for FK
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
            if distance > 450:  # mm
                self.status_bar.configure(text="❌ Target outside workspace bounds!")
                return
            
            if distance < 80:  # mm
                self.status_bar.configure(text="❌ Target too close to base!")
                return
                
            # Use inverse kinematics if available
            if self.kinematics is not None:
                try:
                    # Create target transformation matrix
                    T_target = np.eye(4)
                    T_target[0, 3] = target_x / 1000.0  # Convert to meters
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
                            time.sleep(0.2)  # Small delay between joints
                    
                    self.status_bar.configure(text=f"✅ Moved to ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
                    
                except Exception as e:
                    self.status_bar.configure(text=f"❌ IK failed: {str(e)[:50]}...")
            else:
                # Simple joint-space movement approximation
                self.status_bar.configure(text="⚠️ Simple movement (no IK available)")
                
                # Small movements in joint space
                if abs(dx) > abs(dy):  # Primarily X movement
                    if dx > 0:
                        self.bus.write("Goal_Position", "elbow_flex", current_joints["elbow_flex"] + 2)
                    else:
                        self.bus.write("Goal_Position", "elbow_flex", current_joints["elbow_flex"] - 2)
                else:  # Primarily Y movement
                    if dy > 0:
                        self.bus.write("Goal_Position", "shoulder_pan", current_joints["shoulder_pan"] + 2)
                    else:
                        self.bus.write("Goal_Position", "shoulder_pan", current_joints["shoulder_pan"] - 2)
                
                time.sleep(1)
                self.status_bar.configure(text="✅ Joint movement completed")
        
        except Exception as e:
            self.status_bar.configure(text=f"❌ Movement failed: {str(e)[:50]}...")
    
    def coordinate_z_movement(self, dz, current_joints):
        """Coordinate ALL motors that affect Z-axis for proper height control."""
        
        # Z-axis is affected by ALL arm joints in complex ways:
        # 1. Shoulder lift - Primary Z contributor (direct height change)
        # 2. Elbow flex - Major Z contributor (changes arm reach AND height)
        # 3. Wrist flex - Secondary Z contributor (final gripper positioning)
        # 4. Shoulder pan affects Z through geometry when arm is at angles
        
        print(f"🔧 Coordinating FULL Z-movement: {dz:.1f}mm")
        
        # Get all current positions
        current_shoulder_lift = current_joints["shoulder_lift"]
        current_shoulder_pan = current_joints["shoulder_pan"]
        current_elbow = current_joints["elbow_flex"] 
        current_wrist = current_joints["wrist_flex"]
        
        # Calculate the current arm configuration impact on Z
        # Consider the kinematic chain: base -> shoulder -> elbow -> wrist -> gripper
        
        if dz > 0:  # Moving UP
            print(f"   🔼 Moving UP by {dz:.1f}mm - Coordinating all Z-contributors")
            
            # Strategy for UP movement:
            # 1. Shoulder lift up (primary)
            # 2. Elbow extend slightly (to maintain reach while lifting)  
            # 3. Wrist compensate (to maintain gripper angle)
            
            # Scale movements based on current arm configuration
            # If elbow is very flexed, it contributes more to Z
            elbow_flex_factor = abs(current_elbow) / 90.0  # Normalize elbow contribution
            
            shoulder_delta = 2.5 * (1.0 - elbow_flex_factor * 0.3)  # Less shoulder if elbow very flexed
            elbow_delta = 1.5 * elbow_flex_factor  # More elbow extension if currently flexed
            wrist_delta = -1.0 * (shoulder_delta + elbow_delta) / 2.0  # Compensate for both
            
            print(f"   Shoulder lift: +{shoulder_delta:.1f}° (primary Z)")
            print(f"   Elbow extend: +{elbow_delta:.1f}° (maintain reach)")
            print(f"   Wrist compensate: {wrist_delta:.1f}° (maintain orientation)")
            
        else:  # Moving DOWN
            print(f"   🔽 Moving DOWN by {abs(dz):.1f}mm - Coordinating all Z-contributors")
            
            # Strategy for DOWN movement:
            # 1. Shoulder lift down (primary)
            # 2. Elbow flex more (to control descent)
            # 3. Wrist compensate (to maintain gripper angle)
            
            elbow_flex_factor = abs(current_elbow) / 90.0
            
            shoulder_delta = -2.5 * (1.0 - elbow_flex_factor * 0.3)  # Less shoulder if elbow very flexed
            elbow_delta = -1.5 * (1.0 - elbow_flex_factor)  # More elbow flex if currently extended
            wrist_delta = 1.0 * abs(shoulder_delta + elbow_delta) / 2.0  # Compensate for both
            
            print(f"   Shoulder lower: {shoulder_delta:.1f}° (primary Z)")
            print(f"   Elbow flex: {elbow_delta:.1f}° (control descent)")
            print(f"   Wrist compensate: +{wrist_delta:.1f}° (maintain orientation)")
        
        # Execute coordinated movement with proper sequencing
        try:
            print(f"   🎯 Executing 3-motor coordination...")
            
            # Phase 1: Shoulder movement (primary Z change)
            target_shoulder = current_shoulder_lift + shoulder_delta
            print(f"   Phase 1: Shoulder {current_shoulder_lift:.1f}° → {target_shoulder:.1f}°")
            self.bus.write("Goal_Position", "shoulder_lift", target_shoulder)
            time.sleep(0.8)  # Let shoulder complete first
            
            # Phase 2: Elbow adjustment (secondary Z contributor)
            target_elbow = current_elbow + elbow_delta
            print(f"   Phase 2: Elbow {current_elbow:.1f}° → {target_elbow:.1f}°")
            self.bus.write("Goal_Position", "elbow_flex", target_elbow)
            time.sleep(0.6)  # Let elbow adjust
            
            # Phase 3: Wrist compensation (maintain gripper orientation)
            target_wrist = current_wrist + wrist_delta
            print(f"   Phase 3: Wrist {current_wrist:.1f}° → {target_wrist:.1f}°")
            self.bus.write("Goal_Position", "wrist_flex", target_wrist)
            time.sleep(0.5)  # Final adjustment
            
            # Verify movement
            final_shoulder = self.bus.read("Present_Position", "shoulder_lift", normalize=True)
            final_elbow = self.bus.read("Present_Position", "elbow_flex", normalize=True)
            final_wrist = self.bus.read("Present_Position", "wrist_flex", normalize=True)
            
            print(f"   ✅ Z-coordination complete:")
            print(f"     Shoulder: {current_shoulder_lift:.1f}° → {final_shoulder:.1f}° (Δ{final_shoulder-current_shoulder_lift:+.1f}°)")
            print(f"     Elbow: {current_elbow:.1f}° → {final_elbow:.1f}° (Δ{final_elbow-current_elbow:+.1f}°)")
            print(f"     Wrist: {current_wrist:.1f}° → {final_wrist:.1f}° (Δ{final_wrist-current_wrist:+.1f}°)")
            
        except Exception as e:
            print(f"   ❌ Z-movement coordination failed: {e}")
            raise e
            
    def coordinate_x_movement(self, dx, current_joints):
        """Coordinate motors for X-axis (forward/back) movement."""
        
        # X-axis movement primarily uses elbow, with shoulder/wrist compensation
        current_elbow = current_joints["elbow_flex"]
        current_shoulder = current_joints["shoulder_lift"] 
        current_wrist = current_joints["wrist_flex"]
        
        if dx > 0:  # Moving FORWARD
            # Extend elbow to reach forward
            elbow_delta = 4.0     # Extend elbow (less negative)
            shoulder_delta = -1.0  # Slight shoulder compensation
            wrist_delta = -2.0    # Wrist compensation to maintain orientation
            
        else:  # Moving BACKWARD
            # Flex elbow to retract
            elbow_delta = -4.0    # Flex elbow (more negative)  
            shoulder_delta = 1.0   # Slight shoulder compensation
            wrist_delta = 2.0     # Wrist compensation
        
        print(f"🔧 X-movement: Elbow {elbow_delta:+.1f}°, Shoulder {shoulder_delta:+.1f}°, Wrist {wrist_delta:+.1f}°")
        
        try:
            # Execute coordinated X movement
            self.bus.write("Goal_Position", "elbow_flex", current_elbow + elbow_delta)
            time.sleep(0.6)
            self.bus.write("Goal_Position", "shoulder_lift", current_shoulder + shoulder_delta)  
            time.sleep(0.4)
            self.bus.write("Goal_Position", "wrist_flex", current_wrist + wrist_delta)
            time.sleep(0.4)
            
            print(f"   ✅ X-movement coordination complete")
            
        except Exception as e:
            print(f"   ❌ X-movement failed: {e}")
    
    def advanced_z_movement(self, dz, current_joints):
        """Advanced Z-axis movement considering all kinematic contributions."""
        
        print(f"🔧 ADVANCED Z-Movement: {dz:.1f}mm")
        print("   Analyzing full kinematic chain contribution to Z-axis...")
        
        # Get current joint positions
        shoulder_pan = current_joints["shoulder_pan"]
        shoulder_lift = current_joints["shoulder_lift"] 
        elbow_flex = current_joints["elbow_flex"]
        wrist_flex = current_joints["wrist_flex"]
        
        # Calculate current end-effector height contributors
        # Each joint contributes to final Z position in different ways:
        
        print(f"   Current configuration analysis:")
        print(f"     Shoulder lift: {shoulder_lift:.1f}° (primary Z contributor)")
        print(f"     Elbow flex: {elbow_flex:.1f}° (secondary Z + reach contributor)")
        print(f"     Wrist flex: {wrist_flex:.1f}° (final gripper height)")
        print(f"     Shoulder pan: {shoulder_pan:.1f}° (affects geometry)")
        
        # Calculate movement strategy based on current configuration
        try:
            # Measure current Z position for reference
            current_x, current_y, current_z, _ = self.calculate_base_coordinates(current_joints)
            target_z = current_z + dz
            
            print(f"   Current Z: {current_z:.1f}mm → Target Z: {target_z:.1f}mm")
            
            # Strategy: Use iterative approach with small steps
            step_size = max(1, min(5, abs(dz) / 5))  # 1-5 degree steps
            num_steps = max(1, int(abs(dz) / 10))  # Number of incremental steps
            
            print(f"   Strategy: {num_steps} steps of {step_size:.1f}° each")
            
            for step in range(num_steps):
                print(f"   Step {step+1}/{num_steps}:")
                
                # Read current positions for this step
                current_step_joints = {}
                for motor_name in ["shoulder_lift", "elbow_flex", "wrist_flex"]:
                    current_step_joints[motor_name] = self.bus.read("Present_Position", motor_name, normalize=True)
                
                if dz > 0:  # Moving UP
                    # UP strategy: Coordinate all three Z-affecting joints
                    
                    # 1. Shoulder lift (primary Z contributor)
                    shoulder_step = step_size if step < num_steps-1 else step_size * 0.5
                    new_shoulder = current_step_joints["shoulder_lift"] + shoulder_step
                    
                    # 2. Elbow compensation (maintain X position while lifting)
                    # When shoulder lifts, elbow needs slight extension to maintain reach
                    elbow_step = shoulder_step * 0.3  # 30% of shoulder movement
                    new_elbow = current_step_joints["elbow_flex"] + elbow_step
                    
                    # 3. Wrist compensation (maintain gripper orientation)
                    # Compensate for both shoulder and elbow changes
                    wrist_step = -(shoulder_step + elbow_step) * 0.4
                    new_wrist = current_step_joints["wrist_flex"] + wrist_step
                    
                    print(f"     UP step: Shoulder +{shoulder_step:.1f}°, Elbow +{elbow_step:.1f}°, Wrist {wrist_step:.1f}°")
                    
                else:  # Moving DOWN
                    # DOWN strategy: Careful controlled descent
                    
                    # 1. Shoulder lower (primary)
                    shoulder_step = -step_size if step < num_steps-1 else -step_size * 0.5  
                    new_shoulder = current_step_joints["shoulder_lift"] + shoulder_step
                    
                    # 2. Elbow slight flex (help control descent)
                    elbow_step = shoulder_step * 0.2  # 20% of shoulder movement, same direction
                    new_elbow = current_step_joints["elbow_flex"] + elbow_step
                    
                    # 3. Wrist compensation
                    wrist_step = abs(shoulder_step + elbow_step) * 0.5  # Compensate upward
                    new_wrist = current_step_joints["wrist_flex"] + wrist_step
                    
                    print(f"     DOWN step: Shoulder {shoulder_step:.1f}°, Elbow {elbow_step:.1f}°, Wrist +{wrist_step:.1f}°")
                
                # Execute this step's movements
                self.bus.write("Goal_Position", "shoulder_lift", new_shoulder)
                time.sleep(0.4)
                
                self.bus.write("Goal_Position", "elbow_flex", new_elbow)
                time.sleep(0.3)
                
                self.bus.write("Goal_Position", "wrist_flex", new_wrist) 
                time.sleep(0.3)
                
                # Verify step
                achieved_z = self.calculate_base_coordinates({
                    **current_joints,
                    "shoulder_lift": new_shoulder,
                    "elbow_flex": new_elbow,
                    "wrist_flex": new_wrist
                })[2]
                
                print(f"     Achieved Z: {achieved_z:.1f}mm (step progress: {achieved_z - current_z:.1f}mm)")
            
            print(f"   ✅ Advanced Z-movement complete!")
            
        except Exception as e:
            print(f"   ❌ Advanced Z-movement failed: {e}")
            raise e
    
    def update_workspace_display(self, x_mm, y_mm, z_mm):
        """Update workspace visualization with current robot position."""
        
        # Update side view
        self.workspace_side_canvas.delete("robot_current")
        
        center_x, center_z = 150, 100
        scale = 0.3
        
        # Robot position on side view
        pos_x = center_x + x_mm * scale
        pos_z = center_z - z_mm * scale
        
        # Ensure within canvas bounds
        pos_x = max(10, min(290, pos_x))
        pos_z = max(10, min(190, pos_z))
        
        # Draw current position
        self.workspace_side_canvas.create_oval(pos_x-4, pos_z-4, pos_x+4, pos_z+4,
                                             fill='#e74c3c', outline='#ffffff', width=2,
                                             tags="robot_current")
        self.workspace_side_canvas.create_text(pos_x+10, pos_z, text=f"({x_mm:.0f},{z_mm:.0f})",
                                             font=("Arial", 7), fill='#ecf0f1', tags="robot_current")
        
        # Update top view
        self.workspace_top_canvas.delete("robot_current")
        
        center_top_x, center_top_y = 150, 100
        
        # Robot position on top view
        pos_top_x = center_top_x + y_mm * scale
        pos_top_y = center_top_y - x_mm * scale
        
        # Ensure within canvas bounds
        pos_top_x = max(10, min(290, pos_top_x))
        pos_top_y = max(10, min(190, pos_top_y))
        
        # Draw current position with height indication
        if z_mm > 150:
            color = '#27ae60'  # Green for high
        elif z_mm > 50:
            color = '#f39c12'  # Orange for medium
        else:
            color = '#e74c3c'  # Red for low
        
        self.workspace_top_canvas.create_oval(pos_top_x-5, pos_top_y-5, pos_top_x+5, pos_top_y+5,
                                            fill=color, outline='#ffffff', width=2,
                                            tags="robot_current")
        self.workspace_top_canvas.create_text(pos_top_x+12, pos_top_y, text=f"({x_mm:.0f},{y_mm:.0f})",
                                           font=("Arial", 7), fill='#ecf0f1', tags="robot_current")
    
    def calculate_robot_square(self, shoulder_pan, shoulder_lift):
        """Calculate which chess square robot is over."""
        try:
            # Load chess calibration
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
    
    def update_camera(self):
        """Update camera view."""
        try:
            if hasattr(self, 'camera') and self.camera.is_connected:
                frame = self.camera.read()
                
                # Convert to RGB and resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (320, 240))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo
                
                self.camera_status.configure(text="Status: ✅ Live", fg='#27ae60')
            else:
                self.camera_status.configure(text="Status: ❌ Disconnected", fg='#e74c3c')
                
        except Exception as e:
            self.camera_status.configure(text=f"Status: ❌ Error", fg='#e74c3c')
    
    def update_motors(self):
        """Update motor positions and status."""
        try:
            if hasattr(self, 'bus') and self.bus.is_connected:
                motor_data = {}
                all_good = True
                
                for motor_name in self.all_motors.keys():
                    try:
                        pos = self.bus.read("Present_Position", motor_name, normalize=True)
                        motor_data[motor_name] = {"position": pos, "status": "ok"}
                        
                        # Update display
                        unit = "%" if motor_name == "gripper" else "°"
                        self.motor_labels[motor_name].configure(
                            text=f"{pos:6.1f}{unit}", fg='#27ae60'
                        )
                        self.motor_status_labels[motor_name].configure(
                            text="●", fg='#27ae60'  # Green dot
                        )
                        
                    except Exception as e:
                        motor_data[motor_name] = {"position": None, "status": "error", "error": str(e)}
                        all_good = False
                        
                        # Update display with error
                        self.motor_labels[motor_name].configure(
                            text="ERROR", fg='#e74c3c'
                        )
                        self.motor_status_labels[motor_name].configure(
                            text="●", fg='#e74c3c'  # Red dot
                        )
                
                # Update robot overall status
                if all_good:
                    self.robot_status.configure(text="Robot: ✅ All Motors OK", fg='#27ae60')
                    
                    # Calculate robot base coordinates (forward kinematics)
                    joint_positions = {name: data["position"] for name, data in motor_data.items() if data["position"] is not None}
                    
                    if len(joint_positions) >= 5:  # Need at least 5 joints for FK
                        x_mm, y_mm, z_mm, method = self.calculate_base_coordinates(joint_positions)
                        
                        # Update coordinate display
                        self.coord_x.configure(text=f"X: {x_mm:6.1f} mm")
                        self.coord_y.configure(text=f"Y: {y_mm:6.1f} mm") 
                        self.coord_z.configure(text=f"Z: {z_mm:6.1f} mm")
                        
                        # Calculate distance from base
                        distance = np.sqrt(x_mm**2 + y_mm**2 + z_mm**2)
                        self.distance_label.configure(text=f"Distance: {distance:.1f} mm")
                        
                        # Update workspace status
                        workspace_text = f"Method: {method}\nDistance: {distance:.0f}mm"
                        if distance < 400:
                            workspace_color = '#27ae60'  # Green - in workspace
                        elif distance < 500:
                            workspace_color = '#f39c12'  # Orange - edge of workspace
                        else:
                            workspace_color = '#e74c3c'  # Red - out of reach
                            
                        self.workspace_status.configure(text=workspace_text, fg=workspace_color)
                        
                        # Update workspace visualization
                        self.update_workspace_position(x_mm, y_mm, z_mm)
                        
                        # Update workspace control display
                        self.update_workspace_display(x_mm, y_mm, z_mm)
                        
                        # Update current position in control panel
                        pos_text = f"Current Position:\nX: {x_mm:6.1f} mm\nY: {y_mm:6.1f} mm\nZ: {z_mm:6.1f} mm"
                        self.current_pos_label.configure(text=pos_text)
                        
                        # Update workspace bounds based on current limits
                        bounds_text = f"X: [{x_mm-100:.0f}, {x_mm+100:.0f}] mm\nY: [{y_mm-100:.0f}, {y_mm+100:.0f}] mm\nZ: [{z_mm-50:.0f}, {z_mm+50:.0f}] mm"
                        self.workspace_bounds.configure(text=bounds_text)
                        
                        # Update joint configuration display
                        pan = joint_positions.get("shoulder_pan", 0)
                        lift = joint_positions.get("shoulder_lift", 0)
                        elbow = joint_positions.get("elbow_flex", 0)
                        wrist_flex = joint_positions.get("wrist_flex", 0)
                        wrist_roll = joint_positions.get("wrist_roll", 0)
                        
                        config_text = f"Shoulder: {pan:.1f}°, {lift:.1f}°\nElbow: {elbow:.1f}°\nWrist: {wrist_flex:.1f}°, {wrist_roll:.1f}°"
                        self.joint_config.configure(text=config_text)
                    
                    # Calculate robot position on chess board
                    if "shoulder_pan" in motor_data and "shoulder_lift" in motor_data:
                        square = self.calculate_robot_square(
                            motor_data["shoulder_pan"]["position"],
                            motor_data["shoulder_lift"]["position"]
                        )
                        if square:
                            # Update main position label
                            self.robot_position_label.configure(text=f"Robot at: {square}")
                            
                            # Update detailed position info
                            pan_pos = motor_data["shoulder_pan"]["position"]
                            lift_pos = motor_data["shoulder_lift"]["position"]
                            elbow_pos = motor_data.get("elbow_flex", {}).get("position", 0)
                            
                            details_text = f"Square: {square.upper()}\nPan: {pan_pos:.1f}°\nLift: {lift_pos:.1f}°\nElbow: {elbow_pos:.1f}°"
                            self.robot_details.configure(text=details_text, fg='#27ae60')
                            
                            # Highlight on chess board
                            self.highlight_robot_square(square)
                        else:
                            self.robot_position_label.configure(text="Robot at: Off board")
                            self.robot_details.configure(text="Position: Off board\nMove robot over\nchessboard", fg='#f39c12')
                else:
                    self.robot_status.configure(text="Robot: ⚠️ Motor Issues", fg='#f39c12')
                    
            else:
                self.robot_status.configure(text="Robot: ❌ Disconnected", fg='#e74c3c')
                
        except Exception as e:
            self.robot_status.configure(text="Robot: ❌ Comm Error", fg='#e74c3c')
    
    def highlight_robot_square(self, square):
        """Highlight the square where robot is positioned with enhanced visual."""
        self.draw_chess_board()  # Redraw board
        
        # Calculate square position
        file_idx = ord(square[0]) - ord('a')  # 0-7
        rank_idx = int(square[1]) - 1         # 0-7
        
        # Highlight square (flip rank for display)
        display_row = 7 - rank_idx
        square_size = 30
        
        x1 = file_idx * square_size
        y1 = display_row * square_size
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        # Draw multiple visual indicators for robot position
        
        # 1. Highlight square with pulsing border
        self.chess_canvas.create_rectangle(x1+1, y1+1, x2-1, y2-1, 
                                         outline='#e74c3c', width=4, tags="robot_pos")
        
        # 2. Inner glow effect
        self.chess_canvas.create_rectangle(x1+3, y1+3, x2-3, y2-3, 
                                         outline='#ff6b6b', width=2, tags="robot_pos")
        
        # 3. Robot gripper icon (larger and more prominent)
        self.chess_canvas.create_text(x1 + 15, y1 + 15, text="🤖",
                                    font=("Arial", 14, "bold"), fill='#e74c3c', tags="robot_pos")
        
        # 4. Square label overlay
        self.chess_canvas.create_text(x1 + 15, y1 + 25, text=square.upper(),
                                    font=("Arial", 8, "bold"), fill='#ffffff', tags="robot_pos")
        
        # 5. Position crosshairs
        self.chess_canvas.create_line(x1 + 15, y1, x1 + 15, y2, 
                                    fill='#e74c3c', width=2, tags="robot_pos")
        self.chess_canvas.create_line(x1, y1 + 15, x2, y1 + 15, 
                                    fill='#e74c3c', width=2, tags="robot_pos")
        
        # 6. Direction indicator arrows showing robot orientation
        # Based on current wrist_roll angle, show gripper orientation
        center_x, center_y = x1 + 15, y1 + 15
        
        # Add small directional indicators
        for angle in [0, 90, 180, 270]:  # 4 directions
            rad = np.radians(angle)
            end_x = center_x + 8 * np.cos(rad)
            end_y = center_y + 8 * np.sin(rad)
            
            self.chess_canvas.create_line(center_x, center_y, end_x, end_y,
                                        fill='#3498db', width=1, tags="robot_pos")
    
    def monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        try:
            # Connect systems
            self.bus._connect(handshake=False)
            self.camera.connect()
            
            self.root.after(0, lambda: self.status_bar.configure(
                text="✅ Robot and camera connected - Live monitoring active"))
            
            while self.running:
                # Update camera (30 FPS target)
                self.root.after(0, self.update_camera)
                
                # Update motors (10 FPS - less frequent to avoid overload)
                self.root.after(0, self.update_motors)
                
                time.sleep(0.1)  # 10 FPS update rate
                
        except Exception as e:
            self.root.after(0, lambda: self.status_bar.configure(
                text=f"❌ Monitoring error: {e}"))
        finally:
            try:
                if hasattr(self, 'bus'):
                    self.bus.disconnect()
                if hasattr(self, 'camera'):
                    self.camera.disconnect()
            except:
                pass
    
    def start_monitoring(self):
        """Start monitoring threads."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.status_bar.configure(text="🔄 Starting monitoring systems...")
    
    def stop_monitoring(self):
        """Stop monitoring and close application."""
        self.running = False
        self.root.quit()
    
    def go_home(self):
        """Move robot to home position."""
        try:
            home_path = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/home_joints.npy"
            
            if home_path.exists():
                home_positions = np.load(home_path)
                motor_names = list(self.all_motors.keys())
                
                self.status_bar.configure(text="🏠 Moving to home position...")
                
                # Move to home position
                for i, motor_name in enumerate(motor_names):
                    self.bus.write("Goal_Position", motor_name, float(home_positions[i]))
                    time.sleep(0.5)
                
                self.status_bar.configure(text="✅ Moved to home position")
            else:
                self.status_bar.configure(text="❌ No home position saved")
                
        except Exception as e:
            self.status_bar.configure(text=f"❌ Home move failed: {e}")
    
    def refresh_status(self):
        """Force refresh of all displays."""
        self.status_bar.configure(text="🔄 Refreshing all systems...")
        
        # Force immediate updates
        self.root.after(10, self.update_camera)
        self.root.after(50, self.update_motors)
        
        self.root.after(1000, lambda: self.status_bar.configure(text="✅ Refresh complete"))
    
    def run(self):
        """Start the UI application."""
        try:
            # Configure grid weights for responsive layout
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=1)
            
            main_frame = self.root.winfo_children()[1]  # Get main frame
            main_frame.grid_rowconfigure(0, weight=1)
            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_columnconfigure(1, weight=1)
            main_frame.grid_columnconfigure(2, weight=1)
            main_frame.grid_columnconfigure(3, weight=1)
            
            print("🚀 Starting Chess Robot UI...")
            print("Close the window or click 'Stop' to exit")
            
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("\n🛑 UI interrupted")
        finally:
            self.running = False
            print("👋 Chess Robot UI closed")

def main():
    import argparse
    p = argparse.ArgumentParser(description="Chess Robot Monitoring UI")
    p.add_argument("--port", required=True, help="Robot serial port")
    args = p.parse_args()
    
    try:
        ui = ChessRobotUI(args.port)
        ui.run()
    except Exception as e:
        print(f"❌ UI failed to start: {e}")

if __name__ == "__main__":
    main()
