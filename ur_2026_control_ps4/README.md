# UR5 PS4 Teleop

This repository contains PS4-based teleoperation scripts for one or two UR5 arms over RTDE, plus `multicamera_view.py`.

## Files

- `ps4_rtde_teleop.py`: single-arm teleop with setup target selection, reconnect handling, robot-state pause handling, and Robotiq gripper control
- `ps4_rtde_dual_teleop.py`: dual-arm teleop with left/right/both mode selection, setup target selection, reconnect handling, robot-state pause handling, and synchronized gripper control
- `robotiq_gripper_control.py`: helper used by both teleop scripts to send Robotiq gripper commands through RTDE
- `robotiq_preamble.py`: Robotiq URScript preamble used by `robotiq_gripper_control.py`
- `multicamera_view.py`: camera viewing utility kept in this folder

## Python Dependencies

Install the required packages with:

```bash
pip install ur-rtde pygame
```

## Running

Single arm:

```bash
python ps4_rtde_teleop.py --robot-ip 192.168.1.102
```

Dual arm:

```bash
python ps4_rtde_dual_teleop.py --left-robot-ip 192.168.1.101 --right-robot-ip 192.168.1.102
```

To start without gripper control:

```bash
python ps4_rtde_teleop.py --disable-gripper
python ps4_rtde_dual_teleop.py --disable-gripper
```

## Notes

- Put the robot or robots in Remote Control mode before starting either teleop script.
- Test setup targets and teleop motion in a clear, safe workspace first.
- The hardcoded setup joint and TCP targets are specific to the current lab mounting and may need adjustment for another setup.
