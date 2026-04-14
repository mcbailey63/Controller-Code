import time
import numpy as np
from scipy.spatial.transform import Rotation

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from robotiq_gripper_control import RobotiqGripper

#### SETUP ####
# Establish connections to both robot arms. "Left" and "right" are defined from
# the robot's perspective, not the viewer's perspective.
rtde_c_L = RTDEControl("192.168.1.101")
rtde_r_L = RTDEReceive("192.168.1.101")
rtde_c_R = RTDEControl("192.168.1.102")
rtde_r_R = RTDEReceive("192.168.1.102")

connection_tries = 0
if not rtde_c_L.isConnected():
    while connection_tries < 3:
        rtde_c_L.reconnect()
        time.sleep(0.1)
        if rtde_c_L.isConnected():
            break
        connection_tries += 1

if rtde_c_L.isConnected():
    print("Left robot connection successful!")
else:
    print("Left robot connection not working")
    rtde_c_L.stopScript()

connection_tries = 0
if not rtde_c_R.isConnected():
    while connection_tries < 3:
        rtde_c_R.reconnect()
        time.sleep(0.1)
        if rtde_c_R.isConnected():
            break
        connection_tries += 1

if rtde_c_R.isConnected():
    print("Right robot connection successful!")
else:
    print("Right robot connection not working")
    rtde_c_R.stopScript()

# Prepare grippers
gripper_L = RobotiqGripper(rtde_c_L)
gripper_L.activate()
gripper_L.set_force(50)
gripper_L.set_speed(100)
gripper_L.open()

gripper_R = RobotiqGripper(rtde_c_R)
gripper_R.activate()
gripper_R.set_force(50)
gripper_R.set_speed(100)
gripper_R.open()

try:
    #### COORDINATE TRANSFORMATIONS ####
    # Set offset from robot flange to gripper tip in output flange coordinate frame.
    # Note that if you use a different gripper or tool, you will want to update this!
    tcp_offset = [0, 0, 0.174, 0, 0, 0]
    rtde_c_L.setTcp(tcp_offset)
    rtde_c_R.setTcp(tcp_offset)

    # Choose the top center of the table as the origin of the task coordinate frame.
    # x-axis points towards the right robot arm, y-axis points toward the microwave,
    # and z-axis points up. You may define your own task frame that works for your team;
    # you'll just need to change the rotation matrices between your frame and the robot
    # base frames.

    # LEFT ROBOT ARM
    # Translation in task coordinate frame from left robot base origin to center of table (hand-measured):
    dx_t = 0.090/2 + 0.010 + 0.110 # 1/2 vertical beam width + plate thickness + dist. to robot origin
    dy_t = 0.225/2 + 0.540/2 # half of mounting plate width + half of table depth
    dz_t = -0.753 # height from robot origin to top of table

    # Rotation from task coordinate frame to robot base frame in axis-angle format
    R = np.array([[ 0.707, 0, -0.707],
                  [ 0,    -1,  0],
                  [-0.707, 0, -0.707]])
    rot_base_to_task_L = Rotation.from_matrix(R).as_rotvec().tolist()

    # Rotate translation from task frame coordinates to robot base frame coordinates
    trans_base_to_task_L = np.matmul([dx_t, dy_t, dz_t], R).tolist()

    # Transformation from left robot base frame to task frame
    base_to_task_frame_L = trans_base_to_task_L + rot_base_to_task_L
    task_frame_rot_L = [0, 0, 0] + rot_base_to_task_L # rotation only, for use in re-orienting

    # RIGHT ROBOT ARM
    # Translation in task coordinate frame from right robot base origin to center of table (hand-measured):
    dx_t = -(0.090/2 + 0.010 + 0.110) # 1/2 vertical beam width + plate thickness + dist. to robot origin
    dy_t = 0.225/2 + 0.540/2 # half of mounting plate width + half of table depth
    dz_t = -0.753 # height from robot origin to top of table

    # Rotation from task coordinate frame to robot base frame in axis-angle format
    R = np.array([[ 0.707, 0, 0.707],
                  [ 0,    -1,  0],
                  [ 0.707, 0, -0.707]])
    rot_base_to_task_R = Rotation.from_matrix(R).as_rotvec().tolist()

    # Rotate translation from task frame coordinates to robot base frame coordinates
    trans_base_to_task_R = np.matmul([dx_t, dy_t, dz_t], R).tolist()

    # Transformation from right robot base frame to task frame
    base_to_task_frame_R = trans_base_to_task_R + rot_base_to_task_R
    task_frame_rot_R = [0, 0, 0] + rot_base_to_task_R # rotation only, for use in re-orienting

    # TOP-DOWN CAMERA
    # Translation in task coordinate frame from task frame origin to top-down camera origin (hand-measured):
    dx_task_to_TC = 0.0125 # camera center offset in x
    dy_task_to_TC = -.270 - .126 + .510 + .0425  # half table depth + distance to 8020 base + distance along 8020 to perfboard + camera location on perfboard
    dz_task_to_TC = 1.235 # top of table to camera height
    task_to_TC_tr = [dx_task_to_TC, dy_task_to_TC, dz_task_to_TC]

    # Rotate 180 degrees about x-axis
    task_to_TC_rot = [3.14, 0, 0]

    # Transformation from task frame to camera frame
    task_to_TC = task_to_TC_tr + task_to_TC_rot

    # FRONT-FACING CAMERA
    # TODO: calculate the transformation between the task frame and the 
    # front-facing camera frame. From the camera's perspective, the x-axis
    # is positive to the right, the y-axis is positive downwards, and the 
    # z-axis is positive forwards.

    ###########################################################################
    #### PICK AND PLACE EXAMPLE WITH LEFT ARM ####
    # Move settings
    TCP_vel = 0.1 # end effector velocity [m/s]
    TCP_accel = 0.25 # end effector acceleration [m/s^2]

    # Initial joint position
    # Make sure that it is a safe position for the robot to move into without collsion!
    joint_q_L = np.radians([-48.24, -101.16, -107.03, -99.73, -120.90, -45.00])

    # Move left arm to initial joint position using joint position control
    print("Moving using moveJ to initial position")
    rtde_c_L.moveJ(joint_q_L)
    
    #### OBJECT GRASPING EXAMPLE ####
    # Get object position in top-down camera coordinates
    # This is an example for grabbing the water bottle located directly
    # underneath the top camera. You will need to fetch objects' positions
    # from your perception system in practice.
    obj_pos_topcam = [0.0, 0.0, 1.10]
    obj_rot_topcam = [0, 0, 0] # for now, not worried about object's orientation
    obj_pose_topcam = obj_pos_topcam + obj_rot_topcam

    # Transform object pose from top camera frame to task frame
    obj_pose_task = rtde_c_L.poseTrans(task_to_TC, obj_pose_topcam)

    # Define a pregrasp offset position and orientation from the object to TCP in task frame
    pregrasp_offset_pos = [0, 0, 0.15]
    pregrasp_offset_rot = [0, 0, 0] # for now, not adding any rotation. You will need to figure this out to grasp objects from the side
    pregrasp_offset = pregrasp_offset_pos + pregrasp_offset_rot
    pregrasp_pose_task = [obj_pose_task[i] + pregrasp_offset[i] for i in range(6)]

    # Transform pregrasp pose from task frame to robot base frame
    pregrasp_pose_base = rtde_c_L.poseTrans(base_to_task_frame_L, pregrasp_pose_task)

    # Move robot to pregrasp pose
    gripper_L.open()
    rtde_c_L.moveL(pregrasp_pose_base, TCP_vel, TCP_accel, asynchronous=False)

    # Move TCP to object position and try to grasp it! (Note: for an actual 
    # implementation, you may need to add an offset between the object's center
    # point and the grasp point)
    obj_pose_base = rtde_c_L.poseTrans(base_to_task_frame_L, obj_pose_task)
    rtde_c_L.moveL(obj_pose_base, TCP_vel, TCP_accel, asynchronous=False)
    gripper_L.close()

    # Pick the object up and move it a certain distance in sequence. Note that
    # these are relative moves, but you could specify absolute target poses instead.
    move_up_task = [0, 0, 0.1, 0, 0, 0]
    move_up_task = [obj_pose_task[i] + move_up_task[i] for i in range(6)]
    move_up_base = rtde_c_L.poseTrans(base_to_task_frame_L, move_up_task)
    rtde_c_L.moveL(move_up_base, TCP_vel, TCP_accel, asynchronous=False)

    move_over_task = [-0.2, 0, 0, 0, 0, 0]
    move_over_task = [move_up_task[i] + move_over_task[i] for i in range(6)]
    move_over_base = rtde_c_L.poseTrans(base_to_task_frame_L, move_over_task)
    rtde_c_L.moveL(move_over_base, TCP_vel, TCP_accel, asynchronous=False)

    # Move down until contact is detected, then open gripper to release object
    speed_task = [0, 0, -0.1, 0, 0, 0] # velocity in task frame [m/s, m/s, m/s, rad/s, rad/s, rad/s]
    speed_base = rtde_c_L.poseTrans(task_frame_rot_L, speed_task)
    speed_base[3:6] = [0, 0, 0] # zero out rotational velocity
    rtde_c_L.moveUntilContact(speed_base)
    gripper_L.open()

    # Convert current pose from robot base coordinates to task coordinates
    # TODO: can make this a function to be reused often
    current_pose_base = rtde_r_L.getActualTCPPose()
    task_to_base_L = [-x for x in base_to_task_frame_L]
    T_rot_inv = [0, 0, 0] + [-x for x in base_to_task_frame_L[3:]] # inverse rotation is just negating the rotation vector
    p_neg = [-x for x in base_to_task_frame_L[:3]] + [0, 0, 0]
    task_to_base_L = rtde_c_L.poseTrans(T_rot_inv, p_neg)
    current_pose_task = rtde_c_L.poseTrans(task_to_base_L, current_pose_base)

    # Move straight up in the task frame to clear the object
    move_up_task = [0, 0, 0.2, 0, 0, 0]
    target_pose_task = [current_pose_task[i] + move_up_task[i] for i in range(6)]
    move_up_base = rtde_c_L.poseTrans(base_to_task_frame_L, target_pose_task)
    rtde_c_L.moveL(move_up_base, TCP_vel, TCP_accel, asynchronous=False)

except KeyboardInterrupt as e:
    print("Error:", e)    

# Cleanly end the script
rtde_c_R.stopScript()
rtde_c_L.stopScript()
print("RTDE connection stopped")