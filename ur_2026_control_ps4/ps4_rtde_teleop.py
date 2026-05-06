"""Teleoperate a UR5 over RTDE with a PS4 controller.

Requirements:
- `pip install ur-rtde pygame numpy`
- A PS4 / DualShock 4 controller paired to the computer

Default controls:
- Hold `R1` to enable robot motion (deadman switch)
- Left stick: translate TCP in robot base X / Y
- `R2` / `L2`: translate TCP +Z / -Z
- Right stick X: rotate about Z
- Right stick Y: rotate about Y
- D-pad left / right: rotate about X
- Tap `TRIANGLE`: toggle between TCP control and direct joint control
- Joint mode: left stick X/Y -> joints 1/2, right stick Y/X -> joints 3/4,
  D-pad left/right -> joint 5, `R2`/`L2` -> joint 6
- Hold `CIRCLE`: open gripper gradually
- Hold `SQUARE`: close gripper gradually
- Tap `CROSS`: toggle the selected end effector action
- `OPTIONS`: quit cleanly

These mappings are intentionally simple and conservative. Test with the arm in a
safe area and adjust the signs or speeds below if your controller mapping differs.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import pygame
from robotiq_gripper_control import RobotiqGripper
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_io import RTDEIOInterface as RTDEIO
from rtde_receive import RTDEReceiveInterface as RTDEReceive


CONTROL_PERIOD_S = 1.0 / 125.0
DEFAULT_LINEAR_SPEED = 0.10
DEFAULT_ANGULAR_SPEED = 0.60
DEFAULT_JOINT_SPEED = 0.50
DEFAULT_ACCELERATION = 0.25
DEFAULT_DEADZONE = 0.12
DEFAULT_GRIPPER_OPEN_MM = 50.0
DEFAULT_GRIPPER_CLOSED_MM = 0.0
DEFAULT_GRIPPER_SPEED_MM_PER_S = 60.0
DEFAULT_GRIPPER_FORCE = 50
DEFAULT_GRIPPER_SPEED = 100
GRIPPER_COMMAND_PERIOD_S = 0.08
GRIPPER_MIN_COMMAND_DELTA_MM = 2.5
RECONNECT_RETRY_PERIOD_S = 1.0
RECONNECT_RECREATE_INTERVAL = 5
SETUP_MOVE_SPEED_RAD_PER_S = 0.75
SETUP_MOVE_ACCEL_RAD_PER_S2 = 1.0
SETUP_MOVE_SPEED_M_PER_S = 0.10
SETUP_MOVE_ACCEL_M_PER_S2 = 0.25
USE_SETUP_TCP_POSE = False
TOOL_MODE_GRIPPER = "gripper"
TOOL_MODE_EXTERNAL = "external"
LIMIT_WARN_MARGIN_RAD = math.radians(15.0)
LIMIT_STOP_MARGIN_RAD = math.radians(3.0)
MAX_SAFE_JOINT_SPEED_RAD_PER_S = math.radians(60.0)
MAX_SAFE_CART_SPEED_M_PER_S = 0.20
SLOW_ZONE_MANIPULABILITY = 0.05
WRIST_SINGULAR_THRESH = 0.05
SHOULDER_SINGULAR_THRESH = 0.05
ELBOW_SINGULAR_THRESH = 0.05
MANIPULABILITY_THRESH = 0.01
UR5E_CLAMP_LINK_RADIUS_M = 0.0375
UR5E_CLAMP_FLANGE_RADIUS_M = 0.0375
UR5E_CLAMP_MIN_SURFACE_GAP_M = 0.028
UR5E_CLAMP_STOP_BUFFER_M = 0.012
UR5E_CLAMP_WARN_BUFFER_M = 0.050
UR5E_CLAMP_CENTERLINE_STOP_M = (
    UR5E_CLAMP_LINK_RADIUS_M
    + UR5E_CLAMP_FLANGE_RADIUS_M
    + UR5E_CLAMP_MIN_SURFACE_GAP_M
)
UR5E_CLAMP_COMMAND_STOP_M = (
    UR5E_CLAMP_CENTERLINE_STOP_M + UR5E_CLAMP_STOP_BUFFER_M
)
UR5E_CLAMP_WARN_M = UR5E_CLAMP_CENTERLINE_STOP_M + UR5E_CLAMP_WARN_BUFFER_M

DH_PARAMS = np.array(
    [
        [np.pi / 2.0, 0.0, 0.0892, 0.0],
        [0.0, -0.4251, 0.0, 0.0],
        [0.0, -0.3922, 0.0, 0.0],
        [np.pi / 2.0, 0.0, 0.1093, 0.0],
        [-np.pi / 2.0, 0.0, 0.0948, 0.0],
        [0.0, 0.0, 0.0823, 0.0],
    ],
    dtype=float,
)
JOINT_NAMES = ("Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3")
JOINT_LIMITS = np.array(
    [
        [-2.0 * np.pi, 2.0 * np.pi],
        [-2.0 * np.pi, 2.0 * np.pi],
        [-np.pi, np.pi],
        [-2.0 * np.pi, 2.0 * np.pi],
        [-2.0 * np.pi, 2.0 * np.pi],
        [-2.0 * np.pi, 2.0 * np.pi],
    ],
    dtype=float,
)
RUMBLE_ZONES = (
    (0.005, 1.0, 1.0),
    (0.02, 0.6, 0.8),
    (0.05, 0.2, 0.4),
)

@dataclass(frozen=True)
class ControllerMapping:
    profile_name: str
    button_square: int
    button_cross: int
    button_circle: int
    button_triangle: int
    button_r1: int
    button_share: int
    button_options: int
    button_dpad_left: int
    button_dpad_right: int
    axis_left_x: int
    axis_left_y: int
    axis_right_x: int
    axis_right_y: int
    axis_l2: int
    axis_r2: int


WIRELESS_MAPPING = ControllerMapping(
    profile_name="wireless-ds4",
    button_square=2,
    button_cross=0,
    button_circle=1,
    button_triangle=3,
    button_r1=10,
    button_share=4,
    button_options=6,
    button_dpad_left=13,
    button_dpad_right=14,
    axis_left_x=0,
    axis_left_y=1,
    axis_right_x=2,
    axis_right_y=3,
    axis_l2=4,
    axis_r2=5,
)

WIRED_MAPPING = ControllerMapping(
    profile_name="wired-xinput",
    button_square=2,
    button_cross=0,
    button_circle=1,
    button_triangle=3,
    button_r1=10,
    button_share=4,
    button_options=6,
    button_dpad_left=13,
    button_dpad_right=14,
    axis_left_x=0,
    axis_left_y=1,
    axis_right_x=2,
    axis_right_y=3,
    axis_l2=4,
    axis_r2=5,
)

DEFAULT_MAPPING = WIRELESS_MAPPING

SETUP_JOINTS_DEG = (
    -310.95,
    -113.92,
    75.44,
    269.13,
    -127.80,
    125.89,
)
SETUP_JOINTS_RAD = [math.radians(joint_deg) for joint_deg in SETUP_JOINTS_DEG]
SETUP_TCP_POSE = [
    -246.76e-3,
    200.24e-3,
    -361.85e-3,
    1.319,
    -1.295,
    1.159,
]

# D-pad buttons on this controller show up as ordinary buttons in pygame.
# Hat input is still used as a fallback for other controller drivers.
DPAD_LEFT = -1
DPAD_NEUTRAL = 0
DPAD_RIGHT = 1

VENTION_DUAL_OVERHEAD_ROTATION_LEFT = (
    (0.707, 0.0, -0.707),
    (0.0, -1.0, 0.0),
    (-0.707, 0.0, -0.707),
)
VENTION_DUAL_OVERHEAD_ROTATION_RIGHT = (
    (0.707, 0.0, 0.707),
    (0.0, -1.0, 0.0),
    (0.707, 0.0, -0.707),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate a UR5 over RTDE with a PS4 controller."
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.102",
        help="IP address of the UR robot controller.",
    )
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=DEFAULT_LINEAR_SPEED,
        help="Maximum TCP translation speed in m/s.",
    )
    parser.add_argument(
        "--angular-speed",
        type=float,
        default=DEFAULT_ANGULAR_SPEED,
        help="Maximum TCP rotation speed in rad/s.",
    )
    parser.add_argument(
        "--joint-speed",
        type=float,
        default=DEFAULT_JOINT_SPEED,
        help="Maximum per-joint speed in rad/s while joint mode is active.",
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=DEFAULT_ACCELERATION,
        help="Speed-mode acceleration limit.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=DEFAULT_DEADZONE,
        help="Stick deadzone in the range [0, 1).",
    )
    parser.add_argument(
        "--deadman-button",
        type=int,
        default=DEFAULT_MAPPING.button_r1,
        help="Joystick button index that must be held to enable motion.",
    )
    parser.add_argument(
        "--exit-button",
        type=int,
        default=DEFAULT_MAPPING.button_options,
        help="Joystick button index used to exit cleanly.",
    )
    parser.add_argument(
        "--joint-mode-button",
        type=int,
        default=DEFAULT_MAPPING.button_triangle,
        help="Joystick button index used to toggle TCP/joint control modes.",
    )
    parser.add_argument(
        "--debug-controller",
        action="store_true",
        help="Print live button / axis / hat state to identify your controller mapping.",
    )
    parser.add_argument(
        "--external-tool-output",
        type=int,
        default=0,
        help=(
            "Tool digital output index [0..1] used for a custom tool toggle. "
            "Set to -1 to disable external-tool digital output control."
        ),
    )
    parser.add_argument(
        "--external-tool-button",
        type=int,
        default=DEFAULT_MAPPING.button_cross,
        help="Joystick button index used to toggle the external tool digital output.",
    )
    parser.add_argument(
        "--external-tool-active-high",
        action="store_true",
        default=True,
        help="Drive the configured tool digital output high when the external tool is toggled on.",
    )
    parser.add_argument(
        "--external-tool-active-low",
        dest="external_tool_active_high",
        action="store_false",
        help="Drive the configured tool digital output low when the external tool is toggled on.",
    )
    parser.add_argument(
        "--tool-mode-default",
        choices=(TOOL_MODE_GRIPPER, TOOL_MODE_EXTERNAL),
        default=TOOL_MODE_GRIPPER,
        help=(
            "Active end effector for the CROSS button. "
            "Defaults to the Robotiq gripper."
        ),
    )
    parser.add_argument(
        "--enable-gripper",
        action="store_true",
        default=True,
        help="Initialize and control a Robotiq gripper from the PS4 controller.",
    )
    parser.add_argument(
        "--disable-gripper",
        dest="enable_gripper",
        action="store_false",
        help="Start without initializing the gripper.",
    )
    parser.add_argument(
        "--workspace-frame",
        choices=("base", "vention_left", "vention_right", "vention_auto"),
        default="vention_auto",
        help=(
            "Velocity frame for teleop. "
            "'vention_auto' picks the Vention dual-overhead frame from robot IP."
        ),
    )
    parser.add_argument(
        "--wired-mode",
        action="store_true",
        help=(
            "Use standard wired/XInput-style gamepad indices, such as DS4Windows "
            "gamepad emulation on Windows."
        ),
    )
    parser.add_argument(
        "--disable-kinematic-safety",
        dest="kinematic_safety",
        action="store_false",
        default=True,
        help=(
            "Disable DLS IK singularity slowdown, joint-limit velocity scaling, "
            "and related warnings."
        ),
    )
    parser.add_argument(
        "--disable-rumble",
        dest="rumble_enabled",
        action="store_false",
        default=True,
        help="Disable controller haptic feedback for singularities and joint limits.",
    )
    return parser.parse_args()


def option_was_provided(*option_names: str) -> bool:
    argv = sys.argv[1:]
    for token in argv:
        for option_name in option_names:
            if token == option_name or token.startswith(f"{option_name}="):
                return True
    return False


def resolve_controller_mapping(args: argparse.Namespace) -> ControllerMapping:
    mapping = WIRED_MAPPING if args.wired_mode else WIRELESS_MAPPING

    if not option_was_provided("--deadman-button"):
        args.deadman_button = mapping.button_r1
    if not option_was_provided("--exit-button"):
        args.exit_button = mapping.button_options
    if not option_was_provided("--joint-mode-button"):
        args.joint_mode_button = mapping.button_triangle
    if not option_was_provided("--external-tool-button"):
        args.external_tool_button = mapping.button_cross

    return mapping


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0

    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return scaled if value > 0 else -scaled


def axis(joystick: pygame.joystick.Joystick, index: int) -> float:
    if index < 0 or index >= joystick.get_numaxes():
        return 0.0
    return float(joystick.get_axis(index))


def button(joystick: pygame.joystick.Joystick, index: int) -> bool:
    if index < 0 or index >= joystick.get_numbuttons():
        return False
    return bool(joystick.get_button(index))


def hat_x(joystick: pygame.joystick.Joystick) -> int:
    if joystick.get_numhats() == 0:
        return DPAD_NEUTRAL
    return int(joystick.get_hat(0)[0])


def dpad_x(
    joystick: pygame.joystick.Joystick,
    button_left: int,
    button_right: int,
) -> int:
    left_pressed = button(joystick, button_left)
    right_pressed = button(joystick, button_right)

    if left_pressed and not right_pressed:
        return DPAD_LEFT
    if right_pressed and not left_pressed:
        return DPAD_RIGHT
    if left_pressed and right_pressed:
        return DPAD_NEUTRAL

    return hat_x(joystick)


def hat(joystick: pygame.joystick.Joystick, index: int = 0) -> tuple[int, int]:
    if index >= joystick.get_numhats():
        return (0, 0)
    return tuple(int(v) for v in joystick.get_hat(index))


def pressed_buttons(joystick: pygame.joystick.Joystick) -> list[int]:
    return [i for i in range(joystick.get_numbuttons()) if joystick.get_button(i)]


def active_axes(
    joystick: pygame.joystick.Joystick, threshold: float = 0.20
) -> list[str]:
    active: list[str] = []
    for i in range(joystick.get_numaxes()):
        value = float(joystick.get_axis(i))
        if abs(value) >= threshold:
            active.append(f"{i}:{value:+.2f}")
    return active


def trigger_to_unit_range(raw_value: float) -> float:
    # Many pygame controller mappings report triggers in [-1, 1].
    return max(0.0, min(1.0, 0.5 * (raw_value + 1.0)))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def forward_kinematics(q: np.ndarray) -> list[np.ndarray]:
    transforms = [np.eye(4)]
    for i, (alpha, a, d, offset) in enumerate(DH_PARAMS):
        transforms.append(transforms[-1] @ dh_transform(alpha, a, d, q[i] + offset))
    return transforms


def point_to_segment_distance(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float:
    segment = segment_end - segment_start
    segment_len_sq = float(np.dot(segment, segment))
    if segment_len_sq <= 0.0:
        return float(np.linalg.norm(point - segment_start))

    t = float(np.dot(point - segment_start, segment) / segment_len_sq)
    t = clamp(t, 0.0, 1.0)
    closest = segment_start + t * segment
    return float(np.linalg.norm(point - closest))


def detect_ur5e_clamping(q: np.ndarray) -> dict[str, float | str]:
    transforms = forward_kinematics(q)
    elbow = transforms[2][:3, 3]
    wrist1 = transforms[3][:3, 3]
    flange = transforms[6][:3, 3]
    centerline_distance = point_to_segment_distance(flange, elbow, wrist1)
    surface_gap = (
        centerline_distance
        - UR5E_CLAMP_LINK_RADIUS_M
        - UR5E_CLAMP_FLANGE_RADIUS_M
    )
    margin_to_ur_stop = centerline_distance - UR5E_CLAMP_CENTERLINE_STOP_M

    severity = "clear"
    if centerline_distance < UR5E_CLAMP_COMMAND_STOP_M:
        severity = "stop"
    elif centerline_distance < UR5E_CLAMP_WARN_M:
        severity = "warn"

    return {
        "centerline_distance_m": centerline_distance,
        "surface_gap_m": surface_gap,
        "margin_to_ur_stop_m": margin_to_ur_stop,
        "severity": severity,
    }


def clamping_severity_rank(severity: object) -> int:
    if severity == "stop":
        return 2
    if severity == "warn":
        return 1
    return 0


def jacobian(q: np.ndarray) -> np.ndarray:
    transforms = forward_kinematics(q)
    p_e = transforms[6][:3, 3]
    j = np.zeros((6, 6), dtype=float)
    for i in range(6):
        z_i = transforms[i][:3, 2]
        p_i = transforms[i][:3, 3]
        j[:3, i] = np.cross(z_i, p_e - p_i)
        j[3:, i] = z_i
    return j


def detect_joint_limits(q: np.ndarray) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for i in range(6):
        lower, upper = JOINT_LIMITS[i]
        value = q[i]
        dist_lower = value - lower
        dist_upper = upper - value
        min_dist = min(dist_lower, dist_upper)
        direction = 1 if dist_upper < dist_lower else -1

        if min_dist < LIMIT_WARN_MARGIN_RAD:
            severity = "stop" if min_dist < LIMIT_STOP_MARGIN_RAD else "warn"
            issues.append(
                {
                    "joint": i,
                    "name": JOINT_NAMES[i],
                    "value_deg": math.degrees(value),
                    "margin_deg": math.degrees(min_dist),
                    "severity": severity,
                    "direction": direction,
                }
            )
    return issues


def scale_qdot_for_limits(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    q_dot_safe = q_dot.copy()
    for i in range(6):
        lower, upper = JOINT_LIMITS[i]
        value = q[i]
        dist_lower = value - lower
        dist_upper = upper - value

        if dist_upper < LIMIT_WARN_MARGIN_RAD and q_dot[i] > 0.0:
            if dist_upper < LIMIT_STOP_MARGIN_RAD:
                q_dot_safe[i] = 0.0
            else:
                scale = (dist_upper - LIMIT_STOP_MARGIN_RAD) / (
                    LIMIT_WARN_MARGIN_RAD - LIMIT_STOP_MARGIN_RAD
                )
                q_dot_safe[i] *= np.clip(scale, 0.0, 1.0)
        elif dist_lower < LIMIT_WARN_MARGIN_RAD and q_dot[i] < 0.0:
            if dist_lower < LIMIT_STOP_MARGIN_RAD:
                q_dot_safe[i] = 0.0
            else:
                scale = (dist_lower - LIMIT_STOP_MARGIN_RAD) / (
                    LIMIT_WARN_MARGIN_RAD - LIMIT_STOP_MARGIN_RAD
                )
                q_dot_safe[i] *= np.clip(scale, 0.0, 1.0)
    return q_dot_safe


def scale_qdot_for_ur5e_clamping(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    q_dot_safe = q_dot.copy()
    current = detect_ur5e_clamping(q)
    predicted = detect_ur5e_clamping(q + q_dot_safe * CONTROL_PERIOD_S)
    current_distance = float(current["centerline_distance_m"])
    predicted_distance = float(predicted["centerline_distance_m"])

    if predicted_distance >= UR5E_CLAMP_WARN_M:
        return q_dot_safe

    moving_closer = predicted_distance < current_distance - 1e-6
    if not moving_closer:
        return q_dot_safe

    if predicted_distance <= UR5E_CLAMP_COMMAND_STOP_M:
        q_dot_safe[:] = 0.0
        return q_dot_safe

    scale = (
        (predicted_distance - UR5E_CLAMP_COMMAND_STOP_M)
        / (UR5E_CLAMP_WARN_M - UR5E_CLAMP_COMMAND_STOP_M)
    )
    q_dot_safe *= np.clip(scale, 0.0, 1.0)
    return q_dot_safe


def detect_singularities(q: np.ndarray) -> dict[str, float | bool]:
    transforms = forward_kinematics(q)
    wrist_singular = abs(np.sin(q[4])) < WRIST_SINGULAR_THRESH
    wrist_center = transforms[5][:3, 3]
    dist_to_base_z = float(np.hypot(wrist_center[0], wrist_center[1]))
    shoulder_singular = dist_to_base_z < SHOULDER_SINGULAR_THRESH
    elbow_singular = abs(np.sin(q[2])) < ELBOW_SINGULAR_THRESH
    jv = jacobian(q)[:3, :]
    manipulability = float(np.sqrt(max(0.0, np.linalg.det(jv @ jv.T))))
    return {
        "wrist": wrist_singular,
        "shoulder": shoulder_singular,
        "elbow": elbow_singular,
        "manipulability": manipulability,
        "near_singular": manipulability < MANIPULABILITY_THRESH,
    }


def dls_lambda(
    manipulability: float,
    lambda_max: float = 0.1,
    manip_thresh: float = 0.04,
) -> float:
    if manipulability >= manip_thresh:
        return 0.0
    return lambda_max * (1.0 - manipulability / manip_thresh) ** 2


def cartesian_to_joint_velocity(
    q: np.ndarray,
    x_dot: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    j = jacobian(q)
    sing = detect_singularities(q)
    lam = dls_lambda(float(sing["manipulability"]))
    a = j @ j.T + (lam * lam) * np.eye(6)
    try:
        q_dot = j.T @ np.linalg.solve(a, x_dot)
    except np.linalg.LinAlgError:
        q_dot = j.T @ np.linalg.lstsq(a, x_dot, rcond=None)[0]
    return q_dot, sing


def compute_rumble(
    manipulability: float,
    sing_info: dict[str, float | bool],
    limit_issues: list[dict[str, object]],
) -> tuple[float, float, int]:
    hard_stops = [li for li in limit_issues if li["severity"] == "stop"]
    soft_warns = [li for li in limit_issues if li["severity"] == "warn"]

    if hard_stops:
        return 0.8, 1.0, 100

    for threshold, low, high in RUMBLE_ZONES:
        if manipulability < threshold:
            if sing_info["wrist"]:
                low *= 0.3
            elif sing_info["shoulder"]:
                high *= 0.3
            if soft_warns:
                low = min(1.0, low + 0.2)
                high = min(1.0, high + 0.2)
            return low, high, 60

    if soft_warns:
        intensity = min(1.0, len(soft_warns) * 0.25)
        return intensity * 0.3, intensity * 0.6, 40

    return 0.0, 0.0, 0


def stop_rumble(joystick: pygame.joystick.Joystick) -> None:
    stop = getattr(joystick, "stop_rumble", None)
    if callable(stop):
        stop()


def rumble(
    joystick: pygame.joystick.Joystick,
    low: float,
    high: float,
    duration_ms: int,
) -> bool:
    rumble_method = getattr(joystick, "rumble", None)
    if not callable(rumble_method):
        return False
    return bool(rumble_method(low, high, duration_ms))


def safe_cartesian_joint_speeds(
    q: np.ndarray,
    tcp_twist: list[float],
    *,
    max_joint_speed: float,
    max_cart_speed: float,
) -> tuple[list[float], list[str], dict[str, float | bool], list[dict[str, object]]]:
    warnings: list[str] = []
    x_dot = np.array(tcp_twist, dtype=float)
    linear_norm = float(np.linalg.norm(x_dot[:3]))
    if linear_norm > max_cart_speed:
        x_dot[:3] *= max_cart_speed / linear_norm

    sing = detect_singularities(q)
    manipulability = float(sing["manipulability"])
    if sing["wrist"]:
        warnings.append("WRIST SINGULARITY")
    if sing["shoulder"]:
        warnings.append("SHOULDER SINGULARITY")
    if sing["elbow"]:
        warnings.append("ELBOW SINGULARITY")

    if 0.0 < manipulability < SLOW_ZONE_MANIPULABILITY:
        scale = manipulability / SLOW_ZONE_MANIPULABILITY
        x_dot *= scale
        warnings.append(f"NEAR SINGULAR: speed at {scale:.0%}")

    q_dot, sing = cartesian_to_joint_velocity(q, x_dot)
    max_qdot = float(np.max(np.abs(q_dot)))
    if max_qdot > max_joint_speed:
        q_dot *= max_joint_speed / max_qdot

    limit_issues = detect_joint_limits(q)
    q_dot = scale_qdot_for_limits(q, q_dot)
    commanded_clamp_info = detect_ur5e_clamping(q + q_dot * CONTROL_PERIOD_S)
    q_dot = scale_qdot_for_ur5e_clamping(q, q_dot)
    clamp_info = detect_ur5e_clamping(q + q_dot * CONTROL_PERIOD_S)
    if clamping_severity_rank(commanded_clamp_info["severity"]) > clamping_severity_rank(
        clamp_info["severity"]
    ):
        clamp_info = commanded_clamp_info

    for issue in limit_issues:
        tag = "STOP" if issue["severity"] == "stop" else "WARN"
        warnings.append(
            f"{tag} {issue['name']}: {issue['value_deg']:.1f} deg "
            f"({issue['margin_deg']:.1f} deg from limit)"
        )

    if clamp_info["severity"] != "clear":
        surface_gap_mm = 1000.0 * float(clamp_info["surface_gap_m"])
        margin_mm = 1000.0 * float(clamp_info["margin_to_ur_stop_m"])
        tag = "STOP" if clamp_info["severity"] == "stop" else "WARN"
        warnings.append(
            f"{tag} UR5e CLAMP GUARD: flange/forearm gap {surface_gap_mm:.1f} mm "
            f"({margin_mm:.1f} mm above UR stop)"
        )
        limit_issues.append(
            {
                "severity": clamp_info["severity"],
                "name": "UR5e clamp guard",
            }
        )

    return q_dot.tolist(), warnings, sing, limit_issues


def safe_direct_joint_speeds(
    q: np.ndarray,
    joint_speeds: list[float],
) -> tuple[list[float], list[str], dict[str, float | bool], list[dict[str, object]]]:
    q_dot = np.array(joint_speeds, dtype=float)
    limit_issues = detect_joint_limits(q)
    q_dot = scale_qdot_for_limits(q, q_dot)
    commanded_clamp_info = detect_ur5e_clamping(q + q_dot * CONTROL_PERIOD_S)
    q_dot = scale_qdot_for_ur5e_clamping(q, q_dot)
    clamp_info = detect_ur5e_clamping(q + q_dot * CONTROL_PERIOD_S)
    if clamping_severity_rank(commanded_clamp_info["severity"]) > clamping_severity_rank(
        clamp_info["severity"]
    ):
        clamp_info = commanded_clamp_info
    sing = detect_singularities(q)
    warnings: list[str] = []

    for issue in limit_issues:
        tag = "STOP" if issue["severity"] == "stop" else "WARN"
        warnings.append(
            f"{tag} {issue['name']}: {issue['value_deg']:.1f} deg "
            f"({issue['margin_deg']:.1f} deg from limit)"
        )

    if clamp_info["severity"] != "clear":
        surface_gap_mm = 1000.0 * float(clamp_info["surface_gap_m"])
        margin_mm = 1000.0 * float(clamp_info["margin_to_ur_stop_m"])
        tag = "STOP" if clamp_info["severity"] == "stop" else "WARN"
        warnings.append(
            f"{tag} UR5e CLAMP GUARD: flange/forearm gap {surface_gap_mm:.1f} mm "
            f"({margin_mm:.1f} mm above UR stop)"
        )
        limit_issues.append(
            {
                "severity": clamp_info["severity"],
                "name": "UR5e clamp guard",
            }
        )

    return q_dot.tolist(), warnings, sing, limit_issues


def mat_vec_mul(matrix: tuple[tuple[float, ...], ...], vector: list[float]) -> list[float]:
    return [
        sum(row[i] * vector[i] for i in range(len(vector)))
        for row in matrix
    ]


def infer_workspace_frame(robot_ip: str, workspace_frame: str) -> str:
    if workspace_frame != "vention_auto":
        return workspace_frame
    if robot_ip.endswith(".101"):
        return "vention_left"
    return "vention_right"


def get_workspace_rotation(workspace_frame: str) -> tuple[tuple[float, ...], ...] | None:
    if workspace_frame == "vention_left":
        return VENTION_DUAL_OVERHEAD_ROTATION_LEFT
    if workspace_frame == "vention_right":
        return VENTION_DUAL_OVERHEAD_ROTATION_RIGHT
    return None


def ensure_connected(control: RTDEControl, receive: RTDEReceive) -> None:
    if control.isConnected() and receive.isConnected():
        return

    for _ in range(3):
        if not control.isConnected():
            control.reconnect()
        if not receive.isConnected():
            receive.reconnect()
        time.sleep(0.1)
        if control.isConnected() and receive.isConnected():
            return

    raise RuntimeError("Could not connect to the UR controller over RTDE.")


def ensure_io_connected(io: RTDEIO) -> None:
    is_connected = getattr(io, "isConnected", None)
    reconnect = getattr(io, "reconnect", None)

    # Some ur-rtde builds expose RTDEIOInterface without connection-state helpers.
    # In that case, the constructor is the only connection step we can rely on.
    if not callable(is_connected):
        return

    if is_connected():
        return

    if not callable(reconnect):
        raise RuntimeError(
            "RTDE I/O interface is disconnected and this ur-rtde build does not "
            "provide reconnect support."
        )

    for _ in range(3):
        reconnect()
        time.sleep(0.1)
        if is_connected():
            return

    raise RuntimeError("Could not connect to the UR controller RTDE I/O interface.")


def create_rtde_interfaces(robot_ip: str) -> tuple[RTDEControl, RTDEReceive, RTDEIO]:
    return (RTDEControl(robot_ip), RTDEReceive(robot_ip), RTDEIO(robot_ip))


def create_gripper(
    control: RTDEControl,
    target_mm: float,
    *,
    force: int = DEFAULT_GRIPPER_FORCE,
    speed: int = DEFAULT_GRIPPER_SPEED,
) -> RobotiqGripper:
    gripper = RobotiqGripper(control)
    gripper.activate()
    gripper.set_force(force)
    gripper.set_speed(speed)
    gripper.move_no_wait(target_mm)
    return gripper


def get_robot_state_label(receive: RTDEReceive, method_name: str) -> str | None:
    method = getattr(receive, method_name, None)
    if not callable(method):
        return None

    value = method()
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def robot_requires_pause(receive: RTDEReceive) -> str | None:
    robot_mode = get_robot_state_label(receive, "getRobotMode")
    safety_mode = get_robot_state_label(receive, "getSafetyMode")
    combined = " ".join(label.upper() for label in (robot_mode, safety_mode) if label)

    if not combined:
        return None

    pause_keywords = (
        "PROTECTIVE_STOP",
        "PROTECTIVE STOP",
        "EMERGENCY_STOP",
        "EMERGENCY STOP",
        "FAULT",
        "VIOLATION",
        "RECOVERY",
    )
    if any(keyword in combined for keyword in pause_keywords):
        details = ", ".join(
            f"{name}={label}"
            for name, label in (
                ("robot_mode", robot_mode),
                ("safety_mode", safety_mode),
            )
            if label is not None
        )
        return details

    return None


def set_external_tool_output(io: RTDEIO, output_index: int, signal: bool) -> None:
    if not 0 <= output_index <= 1:
        raise ValueError(
            "Tool digital output index must be 0 or 1. "
            "Use --external-tool-output -1 to disable external-tool control."
        )

    if not io.setToolDigitalOut(output_index, signal):
        raise RuntimeError(f"Failed to set tool digital output {output_index}.")


def disconnect_interface(interface: object | None) -> None:
    if interface is None:
        return

    disconnect = getattr(interface, "disconnect", None)
    if callable(disconnect):
        disconnect()


def connect_robot_session(
    robot_ip: str,
    enable_gripper: bool,
    gripper_target_mm: float,
    *,
    gripper_force: int = DEFAULT_GRIPPER_FORCE,
    gripper_speed: int = DEFAULT_GRIPPER_SPEED,
    external_tool_output: int = -1,
    external_tool_signal: bool = False,
) -> tuple[RTDEControl, RTDEReceive, RTDEIO, RobotiqGripper | None]:
    rtde_c, rtde_r, rtde_io = create_rtde_interfaces(robot_ip)
    ensure_connected(rtde_c, rtde_r)
    ensure_io_connected(rtde_io)

    if external_tool_output >= 0:
        set_external_tool_output(rtde_io, external_tool_output, external_tool_signal)

    gripper = None
    if enable_gripper:
        print("Activating gripper...")
        gripper = create_gripper(
            rtde_c,
            gripper_target_mm,
            force=gripper_force,
            speed=gripper_speed,
        )
        print("Gripper ready.")

    return (rtde_c, rtde_r, rtde_io, gripper)


def close_robot_session(
    control: RTDEControl | None,
    receive: RTDEReceive | None,
    io: RTDEIO | None,
) -> None:
    try:
        if control is not None:
            control.speedStop()
    except Exception:
        pass

    try:
        if control is not None:
            control.stopScript()
    except Exception:
        pass

    try:
        disconnect_interface(control)
    except Exception:
        pass

    try:
        disconnect_interface(receive)
    except Exception:
        pass

    try:
        disconnect_interface(io)
    except Exception:
        pass


def reconnect_robot_session(
    robot_ip: str,
    control: RTDEControl | None,
    receive: RTDEReceive | None,
    io: RTDEIO | None,
    *,
    enable_gripper: bool,
    gripper_target_mm: float,
    attempt_number: int,
    external_tool_output: int,
    external_tool_signal: bool,
) -> tuple[RTDEControl, RTDEReceive, RTDEIO, RobotiqGripper | None]:
    if control is not None and receive is not None and io is not None:
        try:
            ensure_connected(control, receive)
            ensure_io_connected(io)
            if external_tool_output >= 0:
                set_external_tool_output(io, external_tool_output, external_tool_signal)
            gripper = None
            if enable_gripper:
                print("Re-activating gripper after reconnect...")
                gripper = create_gripper(control, gripper_target_mm)
            return (control, receive, io, gripper)
        except Exception:
            pass

    if attempt_number % RECONNECT_RECREATE_INTERVAL == 0:
        close_robot_session(control, receive, io)
        control = None
        receive = None
        io = None

    return connect_robot_session(
        robot_ip,
        enable_gripper,
        gripper_target_mm,
        external_tool_output=external_tool_output,
        external_tool_signal=external_tool_signal,
    )


def wait_for_controller() -> pygame.joystick.Joystick:
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No game controller found. Pair the PS4 controller first.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Using controller: {joystick.get_name()}")
    return joystick


def debug_controller_loop(
    joystick: pygame.joystick.Joystick,
    *,
    exit_button: int,
) -> int:
    print(
        "Controller debug mode active. "
        f"Press button {exit_button} on the controller to exit."
    )
    last_debug_print = 0.0

    while True:
        pygame.event.pump()

        if button(joystick, exit_button):
            print("\nExit requested from controller.")
            return 0

        now = time.time()
        if now - last_debug_print >= 0.10:
            print(
                "\rbuttons="
                f"{pressed_buttons(joystick)} "
                f"axes={active_axes(joystick)} "
                f"hat={hat(joystick)} "
                f"dpad={hat(joystick)}     ",
                end="",
                flush=True,
            )
            last_debug_print = now

        time.sleep(CONTROL_PERIOD_S)


def main() -> int:
    args = parse_args()
    controller_mapping = resolve_controller_mapping(args)
    active_workspace_frame = infer_workspace_frame(args.robot_ip, args.workspace_frame)
    workspace_rotation = get_workspace_rotation(active_workspace_frame)

    pygame.init()
    pygame.joystick.init()
    rtde_c = None
    rtde_r = None
    rtde_io = None
    gripper = None

    try:
        joystick = wait_for_controller()
        print(
            "Controller layout: "
            f"{joystick.get_numaxes()} axes, "
            f"{joystick.get_numbuttons()} buttons, "
            f"{joystick.get_numhats()} hats"
        )

        if args.debug_controller:
            return debug_controller_loop(joystick, exit_button=args.exit_button)

        try:
            rtde_c, rtde_r, rtde_io, gripper = connect_robot_session(
                args.robot_ip,
                args.enable_gripper,
                DEFAULT_GRIPPER_OPEN_MM,
                external_tool_output=args.external_tool_output,
                external_tool_signal=False,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "Please enable remote control on the robot" in message:
                print(
                    "RTDE could not start because Remote Control is disabled on the robot.\n"
                    "On the teach pendant, switch the robot to Remote Control mode, then run this script again.",
                    file=sys.stderr,
                )
                return 1
            raise

        print(f"Connected to robot at {args.robot_ip}")
        print(
            f"Hold button {args.deadman_button} to move. "
            f"Press button {args.exit_button} to exit."
        )
        print(f"Controller profile: {controller_mapping.profile_name}")
        print(
            "Press button "
            f"{args.joint_mode_button} to toggle between TCP and joint control."
        )
        print(
            "Press button "
            f"{args.external_tool_button} to toggle the configured end effector action."
        )
        print(
            f"Press button {controller_mapping.button_share} to move to the configured setup target."
        )
        print(f"Teleop frame: {active_workspace_frame}")
        if args.kinematic_safety:
            print(
                "Kinematic safety: enabled "
                "(DLS IK, singularity slowdown, joint-limit guards)."
            )
        else:
            print("Kinematic safety: disabled.")
        if args.enable_gripper:
            print(
                f"Gripper controls: hold button {controller_mapping.button_circle} to open, "
                f"hold button {controller_mapping.button_square} to close, "
                f"tap button {controller_mapping.button_cross} to toggle full open/close when the "
                "active tool is the gripper."
            )
        if args.external_tool_output >= 0:
            print(
                "External tool control: when the configured end effector is 'external', tap button "
                f"{args.external_tool_button} to toggle tool digital output "
                f"{args.external_tool_output}."
            )

        last_debug_print = 0.0
        prev_toggle_pressed = False
        prev_setup_pressed = False
        prev_joint_mode_pressed = False
        prev_external_tool_pressed = False
        gripper_target_mm = DEFAULT_GRIPPER_OPEN_MM
        last_gripper_command_time = 0.0
        last_gripper_sent_target_mm = gripper_target_mm
        last_loop_time = time.time()
        reconnect_attempts = 0
        disconnected = False
        last_pause_reason = None
        joint_mode_enabled = False
        external_tool_state = False
        rumble_cooldown_s = 0.0
        last_safety_warning_print = 0.0
        last_safety_warning_text = ""
        if args.enable_gripper:
            active_tool_mode = args.tool_mode_default
        elif args.external_tool_output >= 0:
            active_tool_mode = TOOL_MODE_EXTERNAL
        else:
            active_tool_mode = TOOL_MODE_GRIPPER

        while True:
            pygame.event.pump()

            if button(joystick, args.exit_button):
                print("Exit requested from controller.")
                break

            enabled = button(joystick, args.deadman_button)

            left_x = apply_deadzone(
                axis(joystick, controller_mapping.axis_left_x),
                args.deadzone,
            )
            left_y = apply_deadzone(
                axis(joystick, controller_mapping.axis_left_y),
                args.deadzone,
            )
            right_x = apply_deadzone(
                axis(joystick, controller_mapping.axis_right_x),
                args.deadzone,
            )
            right_y = apply_deadzone(
                axis(joystick, controller_mapping.axis_right_y),
                args.deadzone,
            )

            trigger_up = trigger_to_unit_range(
                axis(joystick, controller_mapping.axis_r2)
            )
            trigger_down = trigger_to_unit_range(
                axis(joystick, controller_mapping.axis_l2)
            )
            dpad_roll = dpad_x(
                joystick,
                controller_mapping.button_dpad_left,
                controller_mapping.button_dpad_right,
            )
            open_pressed = button(joystick, controller_mapping.button_circle)
            close_pressed = button(joystick, controller_mapping.button_square)
            toggle_pressed = button(joystick, controller_mapping.button_cross)
            setup_pressed = button(joystick, controller_mapping.button_share)
            joint_mode_pressed = button(joystick, args.joint_mode_button)
            external_tool_pressed = button(joystick, args.external_tool_button)
            now = time.time()
            dt = max(0.0, now - last_loop_time)
            last_loop_time = now
            rumble_cooldown_s = max(0.0, rumble_cooldown_s - dt)

            # Keep the script alive while the operator clears protective or safety stops.
            if disconnected:
                if now - last_debug_print >= RECONNECT_RETRY_PERIOD_S:
                    reconnect_attempts += 1
                    print(
                        "\nWaiting for robot to become available again..."
                        f" reconnect attempt {reconnect_attempts}",
                        flush=True,
                    )
                    try:
                        rtde_c, rtde_r, rtde_io, gripper = reconnect_robot_session(
                            args.robot_ip,
                            rtde_c,
                            rtde_r,
                            rtde_io,
                            enable_gripper=args.enable_gripper,
                            gripper_target_mm=gripper_target_mm,
                            attempt_number=reconnect_attempts,
                            external_tool_output=args.external_tool_output,
                            external_tool_signal=(
                                external_tool_state
                                if args.external_tool_active_high
                                else not external_tool_state
                            ),
                        )
                        disconnected = False
                        reconnect_attempts = 0
                        last_gripper_sent_target_mm = gripper_target_mm
                        last_gripper_command_time = now
                        print("Robot reconnected. Teleop resumed.", flush=True)
                    except RuntimeError as exc:
                        message = str(exc)
                        print(f"Reconnect not ready yet: {message}", flush=True)
                    last_debug_print = now
                time.sleep(CONTROL_PERIOD_S)
                continue

            linear_twist = [
                args.linear_speed * left_x,
                args.linear_speed * -left_y,
                args.linear_speed * (trigger_up - trigger_down),
            ]
            angular_twist = [
                args.angular_speed
                * (
                    DPAD_LEFT
                    if dpad_roll == DPAD_LEFT
                    else DPAD_RIGHT if dpad_roll == DPAD_RIGHT else DPAD_NEUTRAL
                ),
                args.angular_speed * -right_y,
                args.angular_speed * right_x,
            ]

            if workspace_rotation is not None:
                linear_twist = mat_vec_mul(workspace_rotation, linear_twist)
                angular_twist = mat_vec_mul(workspace_rotation, angular_twist)

            # Base-frame twist: [vx, vy, vz, rx, ry, rz]
            tcp_twist = linear_twist + angular_twist
            joint_speeds = [
                args.joint_speed * left_x,
                args.joint_speed * -left_y,
                args.joint_speed * -right_y,
                args.joint_speed * right_x,
                args.joint_speed
                * (
                    DPAD_LEFT
                    if dpad_roll == DPAD_LEFT
                    else DPAD_RIGHT if dpad_roll == DPAD_RIGHT else DPAD_NEUTRAL
                ),
                args.joint_speed * (trigger_up - trigger_down),
            ]

            try:
                pause_reason = robot_requires_pause(rtde_r)
                if pause_reason is not None:
                    if pause_reason != last_pause_reason:
                        print(
                            "\nRobot is connected but not ready for motion."
                            f" Waiting for operator recovery: {pause_reason}",
                            flush=True,
                        )
                        last_pause_reason = pause_reason
                    rtde_c.speedStop()
                    time.sleep(CONTROL_PERIOD_S)
                    continue

                last_pause_reason = None
                actual_q = (
                    np.array(rtde_r.getActualQ(), dtype=float)
                    if args.kinematic_safety
                    else None
                )
                safety_warnings: list[str] = []
                safety_sing: dict[str, float | bool] | None = None
                safety_limit_issues: list[dict[str, object]] = []

                if joint_mode_pressed and not prev_joint_mode_pressed:
                    joint_mode_enabled = not joint_mode_enabled
                    rtde_c.speedStop()
                    print(
                        "\nSwitched to "
                        f"{'joint' if joint_mode_enabled else 'TCP'} control mode.",
                        flush=True,
                    )

                if setup_pressed and not prev_setup_pressed:
                    rtde_c.speedStop()
                    if USE_SETUP_TCP_POSE:
                        print(
                            "\nMoving to hardcoded setup tool pose...",
                            flush=True,
                        )
                        rtde_c.moveL(
                            SETUP_TCP_POSE,
                            SETUP_MOVE_SPEED_M_PER_S,
                            SETUP_MOVE_ACCEL_M_PER_S2,
                        )
                        print("Setup tool pose reached.", flush=True)
                    else:
                        print(
                            "\nMoving to hardcoded setup joint position...",
                            flush=True,
                        )
                        rtde_c.moveJ(
                            SETUP_JOINTS_RAD,
                            SETUP_MOVE_SPEED_RAD_PER_S,
                            SETUP_MOVE_ACCEL_RAD_PER_S2,
                        )
                        print("Setup joint position reached.", flush=True)
                elif (
                    args.external_tool_output >= 0
                    and active_tool_mode == TOOL_MODE_EXTERNAL
                    and external_tool_pressed
                    and not prev_external_tool_pressed
                ):
                    external_tool_state = not external_tool_state
                    output_signal = (
                        external_tool_state
                        if args.external_tool_active_high
                        else not external_tool_state
                    )
                    if rtde_io is None:
                        raise RuntimeError("RTDE I/O interface is not available.")
                    set_external_tool_output(
                        rtde_io, args.external_tool_output, output_signal
                    )
                    print(
                        "\nExternal tool output "
                        f"{args.external_tool_output} set to "
                        f"{'ON' if external_tool_state else 'OFF'}.",
                        flush=True,
                    )
                elif enabled:
                    if args.kinematic_safety and actual_q is not None:
                        if joint_mode_enabled:
                            (
                                safe_speeds,
                                safety_warnings,
                                safety_sing,
                                safety_limit_issues,
                            ) = safe_direct_joint_speeds(actual_q, joint_speeds)
                        else:
                            (
                                safe_speeds,
                                safety_warnings,
                                safety_sing,
                                safety_limit_issues,
                            ) = safe_cartesian_joint_speeds(
                                actual_q,
                                tcp_twist,
                                max_joint_speed=MAX_SAFE_JOINT_SPEED_RAD_PER_S,
                                max_cart_speed=MAX_SAFE_CART_SPEED_M_PER_S,
                            )
                        rtde_c.speedJ(
                            safe_speeds,
                            args.acceleration,
                            CONTROL_PERIOD_S,
                        )
                    elif joint_mode_enabled:
                        rtde_c.speedJ(
                            joint_speeds,
                            args.acceleration,
                            CONTROL_PERIOD_S,
                        )
                    else:
                        rtde_c.speedL(
                            tcp_twist,
                            args.acceleration,
                            CONTROL_PERIOD_S,
                        )
                else:
                    rtde_c.speedStop()
                    if args.rumble_enabled:
                        stop_rumble(joystick)

                if (
                    args.rumble_enabled
                    and args.kinematic_safety
                    and safety_sing is not None
                ):
                    low, high, duration_ms = compute_rumble(
                        float(safety_sing["manipulability"]),
                        safety_sing,
                        safety_limit_issues,
                    )
                    if duration_ms > 0 and rumble_cooldown_s <= 0.0:
                        if rumble(joystick, low, high, duration_ms):
                            rumble_cooldown_s = duration_ms / 1000.0
                    elif duration_ms == 0:
                        stop_rumble(joystick)

                if safety_warnings:
                    safety_warning_text = " | ".join(safety_warnings)
                    if (
                        safety_warning_text != last_safety_warning_text
                        or now - last_safety_warning_print >= 0.50
                    ):
                        print(f"\n{safety_warning_text}", flush=True)
                        last_safety_warning_text = safety_warning_text
                        last_safety_warning_print = now
                else:
                    last_safety_warning_text = ""

                if gripper is not None:
                    if (
                        toggle_pressed
                        and not prev_toggle_pressed
                        and active_tool_mode == TOOL_MODE_GRIPPER
                    ):
                        midpoint_mm = 0.5 * (
                            DEFAULT_GRIPPER_OPEN_MM + DEFAULT_GRIPPER_CLOSED_MM
                        )
                        if gripper_target_mm <= midpoint_mm:
                            gripper_target_mm = DEFAULT_GRIPPER_OPEN_MM
                            print("\nOpening gripper fully...", flush=True)
                        else:
                            gripper_target_mm = DEFAULT_GRIPPER_CLOSED_MM
                            print("\nClosing gripper fully...", flush=True)
                        gripper.move_no_wait(gripper_target_mm)
                        last_gripper_command_time = now
                        last_gripper_sent_target_mm = gripper_target_mm
    
                    if open_pressed != close_pressed:
                        direction = 1.0 if open_pressed else -1.0
                        step_mm = direction * (DEFAULT_GRIPPER_SPEED_MM_PER_S * dt)
                        gripper_target_mm = clamp(
                            gripper_target_mm + step_mm,
                            DEFAULT_GRIPPER_CLOSED_MM,
                            DEFAULT_GRIPPER_OPEN_MM,
                        )
                        target_delta_mm = abs(
                            gripper_target_mm - last_gripper_sent_target_mm
                        )
                        if (
                            target_delta_mm >= GRIPPER_MIN_COMMAND_DELTA_MM
                            and now - last_gripper_command_time >= GRIPPER_COMMAND_PERIOD_S
                        ):
                            gripper.move_no_wait(gripper_target_mm)
                            last_gripper_command_time = now
                            last_gripper_sent_target_mm = gripper_target_mm

                tcp_pose = rtde_r.getActualTCPPose()
            except Exception as exc:
                disconnected = True
                print(
                    "\nLost RTDE connection or robot control session."
                    f" Clear the robot fault on the pendant, then this script will"
                    f" retry automatically.\nReason: {exc}",
                    flush=True,
                )
                try:
                    if rtde_c is not None:
                        rtde_c.speedStop()
                except Exception:
                    pass
                prev_toggle_pressed = toggle_pressed
                prev_setup_pressed = setup_pressed
                prev_joint_mode_pressed = joint_mode_pressed
                prev_external_tool_pressed = external_tool_pressed
                last_debug_print = 0.0
                time.sleep(CONTROL_PERIOD_S)
                continue

            prev_toggle_pressed = toggle_pressed
            prev_setup_pressed = setup_pressed
            prev_joint_mode_pressed = joint_mode_pressed
            prev_external_tool_pressed = external_tool_pressed
            if args.debug_controller and now - last_debug_print >= 0.10:
                print(
                    "\rTCP xyz = "
                    f"({tcp_pose[0]: .3f}, {tcp_pose[1]: .3f}, {tcp_pose[2]: .3f})  "
                    f"{'ENABLED ' if enabled else 'DISABLED'} "
                    f"mode={'JOINT' if joint_mode_enabled else 'TCP'} "
                    f"ee={active_tool_mode} "
                    f"tool={'ON' if external_tool_state else 'OFF'} "
                    f"frame={active_workspace_frame} "
                    f"deadman={args.deadman_button} "
                    f"buttons={pressed_buttons(joystick)} "
                    f"axes={active_axes(joystick)} "
                    f"hat={hat(joystick)} "
                    f"dpad=({controller_mapping.button_dpad_left},{controller_mapping.button_dpad_right})->{dpad_roll:+d}     ",
                    end="",
                    flush=True,
                )
                last_debug_print = now
            elif not args.debug_controller:
                print(
                    "\rTCP xyz = "
                    f"({tcp_pose[0]: .3f}, {tcp_pose[1]: .3f}, {tcp_pose[2]: .3f})  "
                    f"{'ENABLED ' if enabled else 'DISABLED'} "
                    f"mode={'JOINT' if joint_mode_enabled else 'TCP'} "
                    f"ee={active_tool_mode} "
                    f"tool={'ON' if external_tool_state else 'OFF'} "
                    f"frame={active_workspace_frame} "
                    f"deadman={args.deadman_button}",
                    end="",
                    flush=True,
                )
            time.sleep(CONTROL_PERIOD_S)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    finally:
        if "joystick" in locals():
            try:
                stop_rumble(joystick)
            except Exception:
                pass
        close_robot_session(rtde_c, rtde_r, rtde_io)
        pygame.quit()
        print("\nRTDE session closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
