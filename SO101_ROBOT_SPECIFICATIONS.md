# SO-101 Robot Arm Specifications

## 🤖 Robot Overview
- **Name**: SO-101 Follower Arm
- **Designer**: TheRobotStudio and Hugging Face
- **Type**: 6-DOF Serial Manipulator
- **Application**: Chess piece manipulation and precision tasks

## ⚙️ Motor Configuration

| Motor ID | Joint Name | Model | Control Mode | Function |
|----------|------------|-------|--------------|----------|
| 1 | shoulder_pan | STS3215 | Degrees | Left/right rotation of entire arm |
| 2 | shoulder_lift | STS3215 | Degrees | Up/down lift of arm (primary Z-axis) |
| 3 | elbow_flex | STS3215 | Degrees | Elbow bend (secondary Z-axis & reach) |
| 4 | wrist_flex | STS3215 | Degrees | Wrist up/down (gripper orientation) |
| 5 | wrist_roll | STS3215 | Degrees | Wrist rotation (gripper rotation) |
| 6 | gripper | STS3215 | 0-100% | Gripper open/close |

## 📐 Physical Dimensions (Approximate)

Based on kinematic analysis from your robot's movement patterns:

### Link Lengths
- **L0** (Base to Shoulder): ~100mm (estimated)
- **L1** (Shoulder to Elbow): ~150mm 
- **L2** (Elbow to Wrist): ~120mm
- **L3** (Wrist to Gripper): ~80mm

### Total Specifications
- **Maximum Reach**: ~450mm from base center
- **Minimum Reach**: ~80mm (dead zone near base)
- **Optimal Working Radius**: ~200-300mm
- **Total Arm Length**: ~350mm (L1+L2+L3)

## 🔧 Kinematic Chain

```
Base → Shoulder Pan → Shoulder Lift → Elbow Flex → Wrist Flex → Wrist Roll → Gripper
  ↓         ↓             ↓             ↓           ↓           ↓         ↓
Joint 0   Joint 1      Joint 2       Joint 3     Joint 4     Joint 5   End-Effector
```

### Z-Axis Contributors (Height Control)
1. **Shoulder Lift** (Primary): Direct vertical lift of entire arm
2. **Elbow Flex** (Secondary): Changes both reach and height significantly  
3. **Wrist Flex** (Fine): Final gripper height adjustment and orientation

### X-Axis Contributors (Forward/Back)
1. **Elbow Flex** (Primary): Extension/retraction of forearm
2. **Shoulder Lift** (Secondary): Changes forward reach via geometry
3. **Wrist Flex** (Compensation): Maintains gripper orientation

### Y-Axis Contributors (Left/Right)
1. **Shoulder Pan** (Primary): Direct left/right rotation

## 🎯 Workspace Characteristics

### Reachable Volume
- **Shape**: Approximately spherical sector around base
- **Maximum radius**: ~450mm  
- **Minimum radius**: ~80mm (collision with base)
- **Optimal zone**: 200-300mm radius (best accuracy)

### Height Capabilities
- **Maximum height**: ~350mm above base
- **Minimum height**: ~-100mm below base level
- **Chess working height**: ~150-250mm (optimal)

## 🔄 Joint Ranges (Observed from your calibration)

Based on your mixed firmware calibration data:

| Joint | Approximate Range | Notes |
|-------|------------------|-------|
| shoulder_pan | ±90° | Full left-right swing |
| shoulder_lift | -90° to +90° | Full up-down range |  
| elbow_flex | -150° to +150° | Full flex-extend range |
| wrist_flex | -90° to +90° | Gripper orientation control |
| wrist_roll | 0° to 360° | Full rotation capability |
| gripper | 0-100% | 0=open, 100=fully closed |

## 🎮 Chess-Specific Calibration

From your calibration data:

### Chess Coordinate Scaling
- **Files (a→h)**: -8.33°/file (shoulder_pan movement)
- **Ranks (1→8)**: 19.16°/rank (shoulder_lift movement)
- **Reference**: Square a1 as coordinate origin

### Optimal Chess Configuration
- **Height above board**: ~150-200mm
- **Shoulder lift range**: -80° to +20°
- **Pan range**: -60° to +60° (covers full board)
- **Elbow working range**: -60° to -30° (chess height)

## 🔩 Motor Specifications (STS3215)

- **Manufacturer**: Feetech
- **Model**: STS3215  
- **Control**: Serial servo with position feedback
- **Resolution**: 4096 positions (12-bit)
- **Firmware**: Mixed 3.9 and 3.10 (handled by individual control)

## ⚠️ Current Status (Your Robot)

### ✅ Fully Working Motors
- Motor 1 (shoulder_pan): Perfect operation
- Motor 2 (shoulder_lift): Perfect operation  
- Motor 3 (elbow_flex): Working with individual control
- Motor 4 (wrist_flex): Perfect operation
- Motor 5 (wrist_roll): Working with individual control
- Motor 6 (gripper): Working in 20-30% safe range

### 🎯 Proven Capabilities
- **Position accuracy**: 0.2-0.6° repeatability
- **Chess square navigation**: Sub-degree accuracy
- **Coordinated movement**: Multi-joint sequences
- **Workspace coverage**: Full chess board reachable

## 📊 Forward Kinematics Approximation

For precise calculations, the approximate forward kinematics (used in UI):

```python
# Shoulder to elbow vector
L1 = 150  # mm
shoulder_lift_rad = np.radians(shoulder_lift)

# Elbow to wrist vector  
L2 = 120  # mm
total_elbow_angle = shoulder_lift_rad + np.radians(elbow_flex)

# Wrist to gripper vector
L3 = 80   # mm  
total_wrist_angle = total_elbow_angle + np.radians(wrist_flex)

# Calculate end-effector position
reach = L1 * np.cos(shoulder_lift_rad) + L2 * np.cos(total_elbow_angle) + L3 * np.cos(total_wrist_angle)
height = L1 * np.sin(shoulder_lift_rad) + L2 * np.sin(total_elbow_angle) + L3 * np.sin(total_wrist_angle)

# Apply shoulder pan rotation
x_mm = reach * np.cos(np.radians(shoulder_pan))
y_mm = reach * np.sin(np.radians(shoulder_pan))  
z_mm = height + base_height_offset
```

## 🚀 Notes

- **URDF file referenced**: `./SO101/so101_new_calib.urdf` (not currently available)
- **Calibration system**: Uses teach-by-example for chess coordinates
- **Mixed firmware**: Successfully handled with individual motor control
- **Chess optimization**: System specifically tuned for chess piece manipulation

---

*This specification is derived from analysis of your robot's behavior, calibration data, and code implementation. For official specifications, refer to TheRobotStudio documentation.*





