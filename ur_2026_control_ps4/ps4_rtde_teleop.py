"""Teleoperate a UR5 over RTDE with a PS4 controller.

Requirements:
- `pip install ur-rtde pygame`
- A PS4 / DualShock 4 controller paired to the computer

Default controls:
- Hold `R1` to enable robot motion (deadman switch)
- Left stick: translate TCP in robot base X / Y
- `R2` / `L2`: translate TCP +Z / -Z
- Right stick X: rotate about Z
- Right stick Y: rotate about Y
- D-pad left / right: rotate about X
- Hold `CIRCLE`: open gripper gradually
- Hold `SQUARE`: close gripper gradually
- Tap `CROSS`: toggle fully open / fully closed
- `OPTIONS`: quit cleanly

These mappings are intentionally simple and conservative. Test with the arm in a
safe area and adjust the signs or speeds below if your controller mapping differs.
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
DEFAULT_LINEAR_SPEED = 0.10
DEFAULT_ANGULAR_SPEED = 0.60
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

# Common DualShock 4 button indices in pygame.
BUTTON_SQUARE = 2
BUTTON_CROSS = 0
BUTTON_CIRCLE = 1
BUTTON_TRIANGLE = 3
BUTTON_L1 = 9
BUTTON_R1 = 10
BUTTON_SHARE = 4
BUTTON_OPTIONS = 6
BUTTON_DPAD_LEFT = 13
BUTTON_DPAD_RIGHT = 14

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

# Common axis indices in pygame. Triggers are often at axes 4 and 5 on Windows.
AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_RIGHT_X = 2
AXIS_RIGHT_Y = 3
AXIS_L2 = 4
AXIS_R2 = 5

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
        "--exit-button",
        type=int,
        default=BUTTON_OPTIONS,
        help="Joystick button index used to exit cleanly.",
    )
    parser.add_argument(
        "--debug-controller",
        action="store_true",
        help="Print live button / axis / hat state to identify your controller mapping.",
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
    # Many pygame controller mappings report triggers in [-1, 1].
    return max(0.0, min(1.0, 0.5 * (raw_value + 1.0)))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


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


def create_rtde_interfaces(robot_ip: str) -> tuple[RTDEControl, RTDEReceive]:
    return (RTDEControl(robot_ip), RTDEReceive(robot_ip))


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
) -> tuple[RTDEControl, RTDEReceive, RobotiqGripper | None]:
    rtde_c, rtde_r = create_rtde_interfaces(robot_ip)
    ensure_connected(rtde_c, rtde_r)

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

    return (rtde_c, rtde_r, gripper)


def close_robot_session(
    control: RTDEControl | None,
    receive: RTDEReceive | None,
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


def reconnect_robot_session(
    robot_ip: str,
    control: RTDEControl | None,
    receive: RTDEReceive | None,
    *,
    enable_gripper: bool,
    gripper_target_mm: float,
    attempt_number: int,
) -> tuple[RTDEControl, RTDEReceive, RobotiqGripper | None]:
    if control is not None and receive is not None:
        try:
            ensure_connected(control, receive)
            gripper = None
            if enable_gripper:
                print("Re-activating gripper after reconnect...")
                gripper = create_gripper(control, gripper_target_mm)
            return (control, receive, gripper)
        except Exception:
            pass

    if attempt_number % RECONNECT_RECREATE_INTERVAL == 0:
        close_robot_session(control, receive)
        control = None
        receive = None

    return connect_robot_session(robot_ip, enable_gripper, gripper_target_mm)


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


def main() -> int:
    args = parse_args()
    active_workspace_frame = infer_workspace_frame(args.robot_ip, args.workspace_frame)
    workspace_rotation = get_workspace_rotation(active_workspace_frame)

    pygame.init()
    pygame.joystick.init()
    rtde_c = None
    rtde_r = None
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
            rtde_c, rtde_r, gripper = connect_robot_session(
                args.robot_ip,
                args.enable_gripper,
                DEFAULT_GRIPPER_OPEN_MM,
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
        print(
            f"Press button {BUTTON_SHARE} to move to the configured setup target."
        )
        print(f"Teleop frame: {active_workspace_frame}")
        if args.enable_gripper:
            print(
                f"Gripper controls: hold button {BUTTON_CIRCLE} to open, "
                f"hold button {BUTTON_SQUARE} to close, "
                f"tap button {BUTTON_CROSS} to toggle full open/close."
            )

        last_debug_print = 0.0
        prev_toggle_pressed = False
        prev_setup_pressed = False
        gripper_target_mm = DEFAULT_GRIPPER_OPEN_MM
        last_gripper_command_time = 0.0
        last_gripper_sent_target_mm = gripper_target_mm
        last_loop_time = time.time()
        reconnect_attempts = 0
        disconnected = False
        last_pause_reason = None

        while True:
            pygame.event.pump()

            if button(joystick, args.exit_button):
                print("Exit requested from controller.")
                break

            enabled = button(joystick, args.deadman_button)

            left_x = apply_deadzone(axis(joystick, AXIS_LEFT_X), args.deadzone)
            left_y = apply_deadzone(axis(joystick, AXIS_LEFT_Y), args.deadzone)
            right_x = apply_deadzone(axis(joystick, AXIS_RIGHT_X), args.deadzone)
            right_y = apply_deadzone(axis(joystick, AXIS_RIGHT_Y), args.deadzone)

            trigger_up = trigger_to_unit_range(axis(joystick, AXIS_R2))
            trigger_down = trigger_to_unit_range(axis(joystick, AXIS_L2))
            dpad_roll = dpad_x(joystick)
            open_pressed = button(joystick, BUTTON_CIRCLE)
            close_pressed = button(joystick, BUTTON_SQUARE)
            toggle_pressed = button(joystick, BUTTON_CROSS)
            setup_pressed = button(joystick, BUTTON_SHARE)
            now = time.time()
            dt = max(0.0, now - last_loop_time)
            last_loop_time = now

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
                        rtde_c, rtde_r, gripper = reconnect_robot_session(
                            args.robot_ip,
                            rtde_c,
                            rtde_r,
                            enable_gripper=args.enable_gripper,
                            gripper_target_mm=gripper_target_mm,
                            attempt_number=reconnect_attempts,
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
                elif enabled:
                    rtde_c.speedL(tcp_twist, args.acceleration, CONTROL_PERIOD_S)
                else:
                    rtde_c.speedStop()

                if gripper is not None:
                    if toggle_pressed and not prev_toggle_pressed:
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
                last_debug_print = 0.0
                time.sleep(CONTROL_PERIOD_S)
                continue

            prev_toggle_pressed = toggle_pressed
            prev_setup_pressed = setup_pressed
            if args.debug_controller and now - last_debug_print >= 0.10:
                print(
                    "\rTCP xyz = "
                    f"({tcp_pose[0]: .3f}, {tcp_pose[1]: .3f}, {tcp_pose[2]: .3f})  "
                    f"{'ENABLED ' if enabled else 'DISABLED'} "
                    f"frame={active_workspace_frame} "
                    f"deadman={args.deadman_button} "
                    f"buttons={pressed_buttons(joystick)} "
                    f"axes={active_axes(joystick)} "
                    f"hat={hat(joystick)} "
                    f"dpad=({BUTTON_DPAD_LEFT},{BUTTON_DPAD_RIGHT})->{dpad_roll:+d}     ",
                    end="",
                    flush=True,
                )
                last_debug_print = now
            elif not args.debug_controller:
                print(
                    "\rTCP xyz = "
                    f"({tcp_pose[0]: .3f}, {tcp_pose[1]: .3f}, {tcp_pose[2]: .3f})  "
                    f"{'ENABLED ' if enabled else 'DISABLED'} "
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
        close_robot_session(rtde_c, rtde_r)
        pygame.quit()
        print("\nRTDE session closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
