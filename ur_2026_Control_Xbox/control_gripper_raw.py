from robotiq_gripper import RobotiqGripper

# Simple raw-position control for Robotiq 2F-85 (0-255)
# Edit the constants below as needed, then run:
#   python control_gripper_raw.py

# Connection settings
HOST = "192.168.1.104"
PORT = 63352

# Motion settings
POSITION = 125   # 0 = open, 255 = closed
SPEED = 128    # 0-255
FORCE = 128    # 0-255
AUTO_CALIBRATE = True


def main():
    gripper = RobotiqGripper()
    gripper.connect(HOST, PORT)
    try:
        gripper.activate(auto_calibrate=AUTO_CALIBRATE)
        pos, status = gripper.move_and_wait_for_pos(POSITION, SPEED, FORCE)
        print(f"Requested pos={POSITION}, final pos={pos}, status={status.name}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    main()
