"""Teleoperate one or both UR5 arms over RTDE with a PS4 controller.

This script is intentionally separate from `ps4_rtde_teleop.py` so the
single-arm workflow stays untouched.

Default controls:
- Hold `R1` to enable robot motion (deadman switch)
- Left stick: translate TCP in workspace X / Y
- `R2` / `L2`: translate TCP +Z / -Z
- Right stick X: rotate about Z
- Right stick Y: rotate about Y
- D-pad left / right: rotate about X
- Hold `CIRCLE`: open active gripper(s) gradually
- Hold `SQUARE`: close active gripper(s) gradually
- Tap `CROSS`: toggle active gripper(s) fully open / fully closed
- Tap `TRIANGLE`: cycle control target `left -> right -> both`
- `OPTIONS`: quit cleanly
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import pygame
from robotiq_gripper_control import RobotiqGripper
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


CONTROL_PERIOD_S = 1.0 / 125.0
DEFAULT_LINEAR_SPEED = 0.08
DEFAULT_ANGULAR_SPEED = 0.45
DEFAULT_ACCELERATION = 0.20
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

# Common DualShock 4 button indices in pygame.
BUTTON_OPTIONS = 6
BUTTON_SHARE = 4
BUTTON_TRIANGLE = 3
BUTTON_SQUARE = 2
BUTTON_CROSS = 0
BUTTON_CIRCLE = 1
BUTTON_R1 = 10
BUTTON_DPAD_LEFT = 13
BUTTON_DPAD_RIGHT = 14

DPAD_LEFT = -1
DPAD_NEUTRAL = 0
DPAD_RIGHT = 1

AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_RIGHT_X = 2
AXIS_RIGHT_Y = 3
AXIS_L2 = 4
AXIS_R2 = 5

LEFT_ARM_ROTATION = (
    (0.707, 0.0, -0.707),
    (0.0, -1.0, 0.0),
    (-0.707, 0.0, -0.707),
)
RIGHT_ARM_ROTATION = (
    (0.707, 0.0, 0.707),
    (0.0, -1.0, 0.0),
    (0.707, 0.0, -0.707),
)

LEFT_SETUP_JOINTS_DEG = (
    -48.24,
    -101.16,
    -107.03,
    -99.73,
    -120.90,
    135.00,
)
RIGHT_SETUP_JOINTS_DEG = (
    -312.46,
    -69.69,
    101.25,
    -87.30,
    -239.69,
    -135.00,
)
LEFT_SETUP_JOINTS_RAD = [math.radians(joint_deg) for joint_deg in LEFT_SETUP_JOINTS_DEG]
RIGHT_SETUP_JOINTS_RAD = [math.radians(joint_deg) for joint_deg in RIGHT_SETUP_JOINTS_DEG]
LEFT_SETUP_TCP_POSE = [
    0.5783197453752499,
    -0.38249999999999995,
    0.34500089866414657,
    -1.2044556264480804,
    5.052128231793845e-16,
    -2.9015325338529157,
]
RIGHT_SETUP_TCP_POSE = [
    -0.5783197453752499,
    -0.38249999999999995,
    0.34500089866414657,
    1.1998329646178572,
    5.052128231793845e-16,
    -2.9034471336853516,
]

MODE_LEFT = "left"
MODE_RIGHT = "right"
MODE_BOTH = "both"
MODE_SEQUENCE = (MODE_LEFT, MODE_RIGHT, MODE_BOTH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate one or both UR5 arms over RTDE with a PS4 controller."
    )
    parser.add_argument(
        "--left-robot-ip",
        default="192.168.1.101",
        help="IP address of the left UR robot controller.",
    )
    parser.add_argument(
        "--right-robot-ip",
        default="192.168.1.102",
        help="IP address of the right UR robot controller.",
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
        default=BUTTON_R1,
        help="Joystick button index that must be held to enable motion.",
    )
    parser.add_argument(
        "--mode-button",
        type=int,
        default=BUTTON_TRIANGLE,
        help="Joystick button index used to cycle left/right/both control modes.",
    )
    parser.add_argument(
        "--setup-button",
        type=int,
        default=BUTTON_SHARE,
        help="Joystick button index used to move the active arm(s) to the configured setup target.",
    )
    parser.add_argument(
        "--exit-button",
        type=int,
        default=BUTTON_OPTIONS,
        help="Joystick button index used to exit cleanly.",
    )
    parser.add_argument(
        "--enable-gripper",
        action="store_true",
        default=True,
        help="Initialize and control Robotiq grippers from the PS4 controller.",
    )
    parser.add_argument(
        "--disable-gripper",
        dest="enable_gripper",
        action="store_false",
        help="Start without initializing the grippers.",
    )
    parser.add_argument(
        "--debug-controller",
        action="store_true",
        help="Print live button / axis / hat state to identify your controller mapping.",
    )
    return parser.parse_args()


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0

    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return scaled if value > 0 else -scaled


def axis(joystick: pygame.joystick.Joystick, index: int) -> float:
    if index >= joystick.get_numaxes():
        return 0.0
    return float(joystick.get_axis(index))


def button(joystick: pygame.joystick.Joystick, index: int) -> bool:
    if index >= joystick.get_numbuttons():
        return False
    return bool(joystick.get_button(index))


def hat_x(joystick: pygame.joystick.Joystick) -> int:
    if joystick.get_numhats() == 0:
        return DPAD_NEUTRAL
    return int(joystick.get_hat(0)[0])


def dpad_x(joystick: pygame.joystick.Joystick) -> int:
    left_pressed = button(joystick, BUTTON_DPAD_LEFT)
    right_pressed = button(joystick, BUTTON_DPAD_RIGHT)

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
    return max(0.0, min(1.0, 0.5 * (raw_value + 1.0)))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def mat_vec_mul(matrix: tuple[tuple[float, ...], ...], vector: list[float]) -> list[float]:
    return [
        sum(row[i] * vector[i] for i in range(len(vector)))
        for row in matrix
    ]


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


class RobotSession:
    def __init__(
        self,
        name: str,
        robot_ip: str,
        workspace_rotation: tuple[tuple[float, ...], ...],
        setup_joints_rad: list[float],
        setup_tcp_pose: list[float],
        enable_gripper: bool,
    ) -> None:
        self.name = name
        self.robot_ip = robot_ip
        self.workspace_rotation = workspace_rotation
        self.setup_joints_rad = setup_joints_rad
        self.setup_tcp_pose = setup_tcp_pose
        self.enable_gripper = enable_gripper
        self.control: RTDEControl | None = None
        self.receive: RTDEReceive | None = None
        self.gripper: RobotiqGripper | None = None
        self.gripper_target_mm = DEFAULT_GRIPPER_OPEN_MM
        self.last_gripper_command_time = 0.0
        self.last_gripper_sent_target_mm = self.gripper_target_mm
        self.disconnected = False
        self.reconnect_attempts = 0
        self.last_reconnect_print = 0.0
        self.last_pause_reason: str | None = None

    def connect(self) -> None:
        self.control = RTDEControl(self.robot_ip)
        self.receive = RTDEReceive(self.robot_ip)
        ensure_connected(self.control, self.receive)
        self.gripper = None
        if self.enable_gripper:
            print(f"Activating {self.name.lower()} gripper...")
            self.gripper = create_gripper(self.control, self.gripper_target_mm)
            print(f"{self.name} gripper ready.")
        self.disconnected = False
        self.reconnect_attempts = 0
        self.last_reconnect_print = 0.0
        self.last_gripper_sent_target_mm = self.gripper_target_mm
        self.last_gripper_command_time = time.time()
        self.last_pause_reason = None

    def close(self) -> None:
        try:
            if self.control is not None:
                self.control.speedStop()
        except Exception:
            pass

        try:
            if self.control is not None:
                self.control.stopScript()
        except Exception:
            pass

    def mark_disconnected(self, reason: Exception) -> None:
        self.disconnected = True
        print(
            f"\n{self.name} arm lost RTDE connection or robot control session."
            f" Clear the fault on the pendant; this script will retry automatically."
            f"\nReason: {reason}",
            flush=True,
        )
        try:
            if self.control is not None:
                self.control.speedStop()
        except Exception:
            pass
        self.last_reconnect_print = 0.0

    def maybe_reconnect(self, now: float) -> None:
        if not self.disconnected:
            return
        if now - self.last_reconnect_print < RECONNECT_RETRY_PERIOD_S:
            return

        self.reconnect_attempts += 1
        self.last_reconnect_print = now
        print(
            f"\nWaiting for {self.name} arm to become available again..."
            f" reconnect attempt {self.reconnect_attempts}",
            flush=True,
        )

        try:
            if self.control is not None and self.receive is not None:
                ensure_connected(self.control, self.receive)
            else:
                self.connect()
                print(f"{self.name} arm reconnected.", flush=True)
                return

            self.disconnected = False
            self.reconnect_attempts = 0
            print(f"{self.name} arm reconnected.", flush=True)
            return
        except Exception:
            pass

        if self.reconnect_attempts % RECONNECT_RECREATE_INTERVAL == 0:
            self.close()
            self.control = None
            self.receive = None

        try:
            self.connect()
            print(f"{self.name} arm reconnected.", flush=True)
        except Exception as exc:
            print(f"{self.name} arm not ready yet: {exc}", flush=True)

    def speed_stop(self) -> None:
        if self.control is None or self.disconnected:
            return
        try:
            self.control.speedStop()
        except Exception as exc:
            self.mark_disconnected(exc)

    def motion_ready(self) -> bool:
        if self.receive is None or self.control is None or self.disconnected:
            return False

        pause_reason = robot_requires_pause(self.receive)
        if pause_reason is not None:
            if pause_reason != self.last_pause_reason:
                print(
                    f"\n{self.name} arm is connected but not ready for motion."
                    f" Waiting for operator recovery: {pause_reason}",
                    flush=True,
                )
                self.last_pause_reason = pause_reason
            self.speed_stop()
            return False

        self.last_pause_reason = None
        return True

    def send_twist(
        self,
        linear_twist: list[float],
        angular_twist: list[float],
        acceleration: float,
    ) -> None:
        if self.control is None or self.disconnected:
            return

        tcp_twist = (
            mat_vec_mul(self.workspace_rotation, linear_twist)
            + mat_vec_mul(self.workspace_rotation, angular_twist)
        )
        self.control.speedL(tcp_twist, acceleration, CONTROL_PERIOD_S)

    def move_to_setup_target(self) -> None:
        if self.control is None or self.disconnected:
            return

        self.control.speedStop()
        if USE_SETUP_TCP_POSE:
            print(
                f"\nMoving {self.name.lower()} arm to hardcoded setup tool pose...",
                flush=True,
            )
            self.control.moveL(
                self.setup_tcp_pose,
                SETUP_MOVE_SPEED_M_PER_S,
                SETUP_MOVE_ACCEL_M_PER_S2,
            )
            print(f"{self.name} arm setup tool pose reached.", flush=True)
        else:
            print(
                f"\nMoving {self.name.lower()} arm to hardcoded setup joint position...",
                flush=True,
            )
            self.control.moveJ(
                self.setup_joints_rad,
                SETUP_MOVE_SPEED_RAD_PER_S,
                SETUP_MOVE_ACCEL_RAD_PER_S2,
            )
            print(f"{self.name} arm setup joint position reached.", flush=True)

    def update_gripper(
        self,
        *,
        now: float,
        dt: float,
        open_pressed: bool,
        close_pressed: bool,
        toggle_pressed: bool,
        prev_toggle_pressed: bool,
    ) -> None:
        if self.gripper is None or self.disconnected:
            return

        if toggle_pressed and not prev_toggle_pressed:
            midpoint_mm = 0.5 * (DEFAULT_GRIPPER_OPEN_MM + DEFAULT_GRIPPER_CLOSED_MM)
            if self.gripper_target_mm <= midpoint_mm:
                self.gripper_target_mm = DEFAULT_GRIPPER_OPEN_MM
                print(f"\nOpening {self.name.lower()} gripper fully...", flush=True)
            else:
                self.gripper_target_mm = DEFAULT_GRIPPER_CLOSED_MM
                print(f"\nClosing {self.name.lower()} gripper fully...", flush=True)
            self.gripper.move_no_wait(self.gripper_target_mm)
            self.last_gripper_command_time = now
            self.last_gripper_sent_target_mm = self.gripper_target_mm

        if open_pressed != close_pressed:
            direction = 1.0 if open_pressed else -1.0
            step_mm = direction * (DEFAULT_GRIPPER_SPEED_MM_PER_S * dt)
            self.gripper_target_mm = clamp(
                self.gripper_target_mm + step_mm,
                DEFAULT_GRIPPER_CLOSED_MM,
                DEFAULT_GRIPPER_OPEN_MM,
            )
            target_delta_mm = abs(
                self.gripper_target_mm - self.last_gripper_sent_target_mm
            )
            if (
                target_delta_mm >= GRIPPER_MIN_COMMAND_DELTA_MM
                and now - self.last_gripper_command_time >= GRIPPER_COMMAND_PERIOD_S
            ):
                self.gripper.move_no_wait(self.gripper_target_mm)
                self.last_gripper_command_time = now
                self.last_gripper_sent_target_mm = self.gripper_target_mm

    def get_tcp_pose(self) -> list[float] | None:
        if self.receive is None or self.disconnected:
            return None
        return self.receive.getActualTCPPose()


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
                f"dpad=({BUTTON_DPAD_LEFT},{BUTTON_DPAD_RIGHT})->{dpad_x(joystick):+d}     ",
                end="",
                flush=True,
            )
            last_debug_print = now

        time.sleep(CONTROL_PERIOD_S)


def cycle_mode(current_mode: str) -> str:
    current_index = MODE_SEQUENCE.index(current_mode)
    next_index = (current_index + 1) % len(MODE_SEQUENCE)
    return MODE_SEQUENCE[next_index]


def active_sessions_for_mode(
    mode: str,
    left_session: RobotSession,
    right_session: RobotSession,
) -> list[RobotSession]:
    if mode == MODE_LEFT:
        return [left_session]
    if mode == MODE_RIGHT:
        return [right_session]
    return [left_session, right_session]


def main() -> int:
    args = parse_args()

    pygame.init()
    pygame.joystick.init()

    left_session = RobotSession(
        "Left",
        args.left_robot_ip,
        LEFT_ARM_ROTATION,
        LEFT_SETUP_JOINTS_RAD,
        LEFT_SETUP_TCP_POSE,
        args.enable_gripper,
    )
    right_session = RobotSession(
        "Right",
        args.right_robot_ip,
        RIGHT_ARM_ROTATION,
        RIGHT_SETUP_JOINTS_RAD,
        RIGHT_SETUP_TCP_POSE,
        args.enable_gripper,
    )

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
            left_session.connect()
            right_session.connect()
        except RuntimeError as exc:
            message = str(exc)
            if "Please enable remote control on the robot" in message:
                print(
                    "RTDE could not start because Remote Control is disabled on one or both robots.\n"
                    "On the teach pendant, switch both robots to Remote Control mode, then run this script again.",
                    file=sys.stderr,
                )
                return 1
            raise

        print(
            f"Connected to left robot at {args.left_robot_ip}"
            f" and right robot at {args.right_robot_ip}"
        )
        print(
            f"Hold button {args.deadman_button} to move. "
            f"Press button {args.mode_button} to cycle left/right/both. "
            f"Press button {args.setup_button} for the configured setup target. "
            f"Press button {args.exit_button} to exit."
        )
        print("Start mode: both")
        if args.enable_gripper:
            print(
                f"Gripper controls: hold button {BUTTON_CIRCLE} to open, "
                f"hold button {BUTTON_SQUARE} to close, "
                f"tap button {BUTTON_CROSS} to toggle full open/close."
            )

        mode = MODE_BOTH
        prev_mode_pressed = False
        prev_setup_pressed = False
        prev_toggle_pressed = False
        last_debug_print = 0.0
        last_loop_time = time.time()

        while True:
            pygame.event.pump()
            now = time.time()
            dt = max(0.0, now - last_loop_time)
            last_loop_time = now

            if button(joystick, args.exit_button):
                print("Exit requested from controller.")
                break

            left_session.maybe_reconnect(now)
            right_session.maybe_reconnect(now)

            mode_pressed = button(joystick, args.mode_button)
            if mode_pressed and not prev_mode_pressed:
                mode = cycle_mode(mode)
                print(f"\nControl mode changed to: {mode}", flush=True)
            prev_mode_pressed = mode_pressed

            setup_pressed = button(joystick, args.setup_button)
            open_pressed = button(joystick, BUTTON_CIRCLE)
            close_pressed = button(joystick, BUTTON_SQUARE)
            toggle_pressed = button(joystick, BUTTON_CROSS)

            enabled = button(joystick, args.deadman_button)

            left_x = apply_deadzone(axis(joystick, AXIS_LEFT_X), args.deadzone)
            left_y = apply_deadzone(axis(joystick, AXIS_LEFT_Y), args.deadzone)
            right_x = apply_deadzone(axis(joystick, AXIS_RIGHT_X), args.deadzone)
            right_y = apply_deadzone(axis(joystick, AXIS_RIGHT_Y), args.deadzone)
            trigger_up = trigger_to_unit_range(axis(joystick, AXIS_R2))
            trigger_down = trigger_to_unit_range(axis(joystick, AXIS_L2))
            dpad_roll = dpad_x(joystick)

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

            target_sessions = active_sessions_for_mode(mode, left_session, right_session)
            motion_ready_sessions = [
                session for session in target_sessions if session.motion_ready()
            ]

            for session in (left_session, right_session):
                if session not in target_sessions or not enabled or setup_pressed:
                    session.speed_stop()

            if setup_pressed and not prev_setup_pressed:
                for session in motion_ready_sessions:
                    try:
                        session.move_to_setup_target()
                    except Exception as exc:
                        session.mark_disconnected(exc)
            elif enabled:
                for session in motion_ready_sessions:
                    try:
                        session.send_twist(linear_twist, angular_twist, args.acceleration)
                    except Exception as exc:
                        session.mark_disconnected(exc)

            for session in target_sessions:
                try:
                    session.update_gripper(
                        now=now,
                        dt=dt,
                        open_pressed=open_pressed,
                        close_pressed=close_pressed,
                        toggle_pressed=toggle_pressed,
                        prev_toggle_pressed=prev_toggle_pressed,
                    )
                except Exception as exc:
                    session.mark_disconnected(exc)

            left_pose = None
            right_pose = None
            try:
                left_pose = left_session.get_tcp_pose()
            except Exception as exc:
                left_session.mark_disconnected(exc)

            try:
                right_pose = right_session.get_tcp_pose()
            except Exception as exc:
                right_session.mark_disconnected(exc)

            if args.debug_controller and now - last_debug_print >= 0.10:
                print(
                    "\r"
                    f"mode={mode} "
                    f"{'ENABLED ' if enabled else 'DISABLED'} "
                    f"L={'disc' if left_pose is None else f'({left_pose[0]: .3f}, {left_pose[1]: .3f}, {left_pose[2]: .3f})'} "
                    f"R={'disc' if right_pose is None else f'({right_pose[0]: .3f}, {right_pose[1]: .3f}, {right_pose[2]: .3f})'} "
                    f"buttons={pressed_buttons(joystick)} "
                    f"axes={active_axes(joystick)} "
                    f"hat={hat(joystick)}     ",
                    end="",
                    flush=True,
                )
                last_debug_print = now
            elif not args.debug_controller:
                left_status = "disc" if left_pose is None else f"({left_pose[0]: .3f}, {left_pose[1]: .3f}, {left_pose[2]: .3f})"
                right_status = "disc" if right_pose is None else f"({right_pose[0]: .3f}, {right_pose[1]: .3f}, {right_pose[2]: .3f})"
                print(
                    "\r"
                    f"mode={mode} "
                    f"{'ENABLED ' if enabled else 'DISABLED'} "
                    f"L={left_status} "
                    f"R={right_status}",
                    end="",
                    flush=True,
                )

            prev_setup_pressed = setup_pressed
            prev_toggle_pressed = toggle_pressed
            time.sleep(CONTROL_PERIOD_S)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    finally:
        left_session.close()
        right_session.close()
        pygame.quit()
        print("\nRTDE sessions closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
